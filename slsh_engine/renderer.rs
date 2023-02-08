use std::default::Default;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::Display;
use std::mem::size_of;
use std::ptr;
use std::str::FromStr;

use ash::extensions::khr::{Surface, Swapchain};
use ash::vk;

use crate::window::Window;

macro_rules! include_shader {
    ($name:literal) => {
        include_bytes!(concat!("../target/shaders/", $name, ".spv"))
    };
}

const REQ_DEVICE_EXTENSIONS: &[&str] = &[
    "VK_KHR_swapchain",
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    "VK_KHR_portability_subset",
];
const REQ_VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];
const API_VER_MAJOR: u32 = 1;
const API_VER_MINOR: u32 = 0;
const API_VER_PATCH: u32 = 0;

const FRAMES_IN_FLIGHT: usize = 2;

trait CheckVkError<T> {
    fn check_err(self, action: &'static str) -> T;
}

pub struct Renderer {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_resolution: vk::Extent2D,
    phys_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    index_count: u32,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    is_rendering: Vec<vk::Fence>,
    current_frame: usize,
}

#[derive(Default, Clone)]
struct QueueFamilyIndices {
    graphics: Option<u32>,
    compute: Option<u32>,
    transfer: Option<u32>,
    present: Option<u32>,
}

#[derive(Clone)]
struct PhysDeviceInfo {
    phys_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    queue_family_indices: QueueFamilyIndices,
    extensions: Vec<vk::ExtensionProperties>,
}

#[repr(C, packed)]
#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

impl Renderer {
    pub unsafe fn new(app_name: &'static str, window: &Window) -> Self {
        let entry = ash::Entry::linked();
        let instance = create_instance(app_name, &entry, window);
        let surface = window.create_surface(&instance);
        let surface_loader = Surface::new(&entry, &instance);
        let phys_device_info = pick_phys_device(&instance, surface, &surface_loader);
        let phys_device = phys_device_info.phys_device;
        let device = create_logical_device(&instance, &phys_device_info);
        let gfx_queue_idx = phys_device_info.queue_family_indices.graphics.unwrap();
        let present_queue_idx = phys_device_info.queue_family_indices.present.unwrap();
        let graphics_queue = device.get_device_queue(gfx_queue_idx, 0);
        let present_queue = device.get_device_queue(present_queue_idx, 0);
        let surface_format = choose_swapchain_format(phys_device, &surface_loader, surface);
        let surface_capabilities = get_surface_capabilities(phys_device, &surface_loader, surface);
        let surface_resolution = choose_swapchain_extent(window, &surface_capabilities);
        let swapchain_loader = Swapchain::new(&instance, &device);
        let swapchain = create_swapchain(
            phys_device,
            surface,
            &surface_loader,
            &surface_capabilities,
            surface_format,
            surface_resolution,
            &swapchain_loader,
        );
        let swapchain_images = get_swapchain_images(&swapchain_loader, swapchain);
        let swapchain_image_views = create_image_views(&device, surface_format, &swapchain_images);
        let command_pool = create_command_pool(&device, gfx_queue_idx);
        let command_buffers =
            create_command_buffers(&device, command_pool, FRAMES_IN_FLIGHT as u32);
        let render_pass = create_render_pass(&device, surface_format.format);
        let pipeline_layout = create_pipeline_layout(&device);
        let pipeline =
            create_graphics_pipeline(&device, surface_resolution, render_pass, pipeline_layout);
        let framebuffers =
            create_framebuffers(&device, &swapchain_image_views, surface_resolution, render_pass);

        let device_mem_properties = instance.get_physical_device_memory_properties(phys_device);

        let vertices = [
            Vertex {
                pos: [-0.5, -0.5],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                pos: [0.5, -0.5],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                pos: [0.5, 0.5],
                color: [0.0, 0.0, 1.0],
            },
            Vertex {
                pos: [-0.5, 0.5],
                color: [1.0, 1.0, 1.0],
            },
        ];

        let (vertex_buffer, vertex_buffer_memory) = create_buffer_of_type(
            &device,
            &device_mem_properties,
            command_pool,
            graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        );

        let indices = [0, 1, 2, 2, 3, 0];

        let (index_buffer, index_buffer_memory) = create_buffer_of_type(
            &device,
            &device_mem_properties,
            command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );

        let index_count = indices.len() as u32;

        let (image_available, render_finished, is_rendering) = create_sync_objects(&device);

        Self {
            entry,
            instance,
            surface_loader,
            surface,
            surface_format,
            surface_resolution,
            phys_device,
            device,
            graphics_queue,
            present_queue,
            swapchain_loader,
            swapchain,
            swapchain_image_views,
            command_pool,
            command_buffers,
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffers,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count,
            image_available,
            render_finished,
            is_rendering,
            current_frame: 0,
        }
    }

    fn record_commands_to_buffer(
        &self,
        cmd_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
    ) {
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            ..Default::default()
        };

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let render_pass_info = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            render_pass: self.render_pass,
            framebuffer,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.surface_resolution,
            },
            clear_value_count: 1,
            p_clear_values: &clear_color,
            ..Default::default()
        };

        unsafe {
            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .check_err("reset cmd buffer");

            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .check_err("begin recording to command buffer");

            self.device.cmd_begin_render_pass(
                cmd_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            self.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            self.device.cmd_bind_vertex_buffers(cmd_buffer, 0, &[self.vertex_buffer], &[0]);

            self.device.cmd_bind_index_buffer(
                cmd_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );

            self.device.cmd_draw_indexed(cmd_buffer, self.index_count, 1, 0, 0, 0);

            self.device.cmd_end_render_pass(cmd_buffer);

            self.device.end_command_buffer(cmd_buffer).check_err("end command buffer recording");
        }
    }

    pub fn present(&mut self) {
        let timeout = u64::MAX;

        let command_buffer = self.command_buffers[self.current_frame];
        let image_available = self.image_available[self.current_frame];
        let render_finished = self.render_finished[self.current_frame];
        let is_rendering = self.is_rendering[self.current_frame];

        let image_index = unsafe {
            self.device
                .wait_for_fences(&[is_rendering], true, timeout)
                .check_err("wait for fences");

            let (image_index, _out_of_date) = self
                .swapchain_loader
                .acquire_next_image(self.swapchain, timeout, image_available, vk::Fence::null())
                .check_err("acquire next image");

            // if out_of_date {
            //     self.recreate_swapchain();
            //     return;
            // }

            self.device.reset_fences(&[is_rendering]).check_err("reset fences");

            image_index
        };

        self.record_commands_to_buffer(command_buffer, self.framebuffers[image_index as usize]);

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            wait_semaphore_count: 1,
            p_wait_semaphores: &image_available,
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 1,
            p_signal_semaphores: &render_finished,
            ..Default::default()
        };

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], is_rendering)
                .check_err("submit to draw queue");
        }

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            wait_semaphore_count: 1,
            p_wait_semaphores: &render_finished,
            swapchain_count: 1,
            p_swapchains: &self.swapchain,
            p_image_indices: &image_index,
            ..Default::default()
        };

        unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
                .check_err("queue image for presentation");
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    unsafe fn cleanup_swapchain(&self) {
        self.device.device_wait_idle().unwrap();

        for fb in &self.framebuffers {
            self.device.destroy_framebuffer(*fb, None);
        }

        self.device.destroy_pipeline(self.pipeline, None);
        self.device.destroy_pipeline_layout(self.pipeline_layout, None);
        self.device.destroy_render_pass(self.render_pass, None);
        self.device.free_command_buffers(self.command_pool, &self.command_buffers);
        for image_view in &self.swapchain_image_views {
            self.device.destroy_image_view(*image_view, None);
        }
        self.swapchain_loader.destroy_swapchain(self.swapchain, None);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            for sem in &self.image_available {
                self.device.destroy_semaphore(*sem, None);
            }

            for sem in &self.render_finished {
                self.device.destroy_semaphore(*sem, None);
            }

            for fence in &self.is_rendering {
                self.device.destroy_fence(*fence, None);
            }

            self.cleanup_swapchain();

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Vertex {
    fn get_binding_desc() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_desc() -> [vk::VertexInputAttributeDescription; 2] {
        let pos_offset = 0;
        let col_offset = 2 * size_of::<f32>() as u32;

        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: pos_offset,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: col_offset,
            },
        ]
    }
}

impl<T> CheckVkError<T> for Option<T> {
    fn check_err(self, action: &'static str) -> T {
        match self {
            Some(t) => t,
            None => panic!("Failed to {}", action),
        }
    }
}

impl<T, E: Display> CheckVkError<T> for Result<T, E> {
    fn check_err(self, action: &'static str) -> T {
        match self {
            Ok(t) => t,
            Err(e) => panic!("Failed to {}: err = {}", action, e),
        }
    }
}

fn create_instance(app_name: &'static str, entry: &ash::Entry, window: &Window) -> ash::Instance {
    let app_cstring = CString::new(app_name).check_err("convert app_name to CString");
    let app_cstr = app_cstring.as_c_str();

    let engine_name = CStr::from_bytes_with_nul(b"slsh_engine\0").unwrap();

    let eng_ver_major = u32::from_str(env!("CARGO_PKG_VERSION_MAJOR")).unwrap();
    let eng_ver_minor = u32::from_str(env!("CARGO_PKG_VERSION_MINOR")).unwrap();
    let eng_ver_patch = u32::from_str(env!("CARGO_PKG_VERSION_PATCH")).unwrap();
    let engine_version = vk::make_api_version(0, eng_ver_major, eng_ver_minor, eng_ver_patch);

    let api_version = vk::make_api_version(0, API_VER_MAJOR, API_VER_MINOR, API_VER_PATCH);

    let app_info = vk::ApplicationInfo::builder()
        .application_name(app_cstr)
        .application_version(0)
        .engine_name(engine_name)
        .engine_version(engine_version)
        .api_version(api_version);

    let req_layers_owned = convert_to_strings(REQ_VALIDATION_LAYERS);
    let req_layers_cstrs = convert_to_c_strs(&req_layers_owned);
    let req_layers_cptrs = convert_to_c_ptrs(&req_layers_cstrs);

    let req_exts_owned = window.get_required_extensions();
    let req_exts_cstrs = convert_to_c_strs(&req_exts_owned);

    #[allow(unused_mut)]
    let mut req_exts_cptrs = convert_to_c_ptrs(&req_exts_cstrs);

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        req_exts_cptrs.push(vk::KhrPortabilityEnumerationFn::name().as_ptr());
        req_exts_cptrs.push(vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
    }

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&req_layers_cptrs)
        .enabled_extension_names(&req_exts_cptrs)
        .flags(create_flags);

    unsafe { entry.create_instance(&create_info, None) }.check_err("create instance")
}

fn convert_to_strings(strs: &[&str]) -> Vec<String> {
    strs.iter().map(std::string::ToString::to_string).collect()
}

fn convert_to_c_strs(strings: &[String]) -> Vec<CString> {
    strings
        .iter()
        .map(|string| CString::new(string.clone()).check_err("convert to CString"))
        .collect()
}

fn convert_to_c_ptrs(cstrings: &[CString]) -> Vec<*const c_char> {
    cstrings.iter().map(|cstring| cstring.as_ptr()).collect()
}

unsafe fn pick_phys_device(
    instance: &ash::Instance,
    surface: vk::SurfaceKHR,
    surface_loader: &Surface,
) -> PhysDeviceInfo {
    let phys_devices = instance.enumerate_physical_devices().check_err("get physical devices");
    let mut phys_device_infos =
        gather_phys_device_infos(instance, surface, surface_loader, &phys_devices);

    assert!(!phys_device_infos.is_empty(), "No suitable devices found");

    phys_device_infos.sort_by_key(|d| device_type_to_priority(d.properties.device_type));

    phys_device_infos[0].clone()
}

unsafe fn gather_phys_device_infos(
    instance: &ash::Instance,
    surface: vk::SurfaceKHR,
    surface_loader: &Surface,
    phys_devices: &[vk::PhysicalDevice],
) -> Vec<PhysDeviceInfo> {
    let mut phys_device_infos = Vec::with_capacity(phys_devices.len());

    println!("Physical devices:");

    for device_ref in phys_devices {
        let phys_device = *device_ref;
        let properties = instance.get_physical_device_properties(phys_device);
        let queue_family_indices =
            get_queue_family_indices(instance, phys_device, surface, surface_loader);
        let supports_required_queues =
            queue_family_indices.graphics.is_some() && queue_family_indices.present.is_some();
        let extensions = instance
            .enumerate_device_extension_properties(phys_device)
            .check_err("enumerate device extensions");

        let name = CStr::from_ptr(properties.device_name.as_ptr()).to_str().unwrap();
        println!("\t{}", name);

        if supports_required_queues && supports_required_extensions(&extensions) {
            let info = PhysDeviceInfo {
                phys_device,
                properties,
                queue_family_indices,
                extensions,
            };

            phys_device_infos.push(info);
        }
    }

    phys_device_infos
}

fn supports_required_extensions(exts: &[vk::ExtensionProperties]) -> bool {
    let req_device_exts = convert_to_strings(REQ_DEVICE_EXTENSIONS);
    let req_exts = convert_to_c_strs(&req_device_exts);

    let mut support_found = Vec::with_capacity(req_exts.len());
    support_found.resize(req_exts.len(), false);

    for (i, req_ext) in req_exts.into_iter().enumerate() {
        for ext in exts {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };

            if name == req_ext.as_c_str() {
                support_found[i] = true;
            }
        }
    }

    support_found.into_iter().all(|found| found)
}

fn device_type_to_priority(type_: vk::PhysicalDeviceType) -> i32 {
    match type_ {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 3,
        vk::PhysicalDeviceType::CPU => 4,
        _ => 5,
    }
}

fn choose_swapchain_format(
    phys_device: vk::PhysicalDevice,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
) -> vk::SurfaceFormatKHR {
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(phys_device, surface) }
            .check_err("get surface formats");

    formats[0]
}

fn get_surface_capabilities(
    phys_device: vk::PhysicalDevice,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
) -> vk::SurfaceCapabilitiesKHR {
    unsafe { surface_loader.get_physical_device_surface_capabilities(phys_device, surface) }
        .check_err("get surface capabilities")
}

fn choose_swapchain_extent(
    window: &Window,
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }

    vk::Extent2D {
        width: window.width(),
        height: window.height(),
    }
}

fn get_queue_family_indices(
    instance: &ash::Instance,
    phys_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &Surface,
) -> QueueFamilyIndices {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(phys_device) };

    let mut families = QueueFamilyIndices::default();

    for (i, f) in queue_families.iter().enumerate() {
        let idx = i.try_into().unwrap();
        let opt = Some(idx);

        if f.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            families.graphics = opt;
        }

        if f.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            families.compute = opt;
        }

        if f.queue_flags.contains(vk::QueueFlags::TRANSFER) {
            families.transfer = opt;
        }

        let present_support = unsafe {
            surface_loader
                .get_physical_device_surface_support(phys_device, idx, surface)
                .check_err("get surface support")
        };

        if present_support {
            families.present = opt;
        }
    }

    families
}

fn create_logical_device(instance: &ash::Instance, info: &PhysDeviceInfo) -> ash::Device {
    let mut unique_families = vec![
        info.queue_family_indices.graphics.unwrap(),
        info.queue_family_indices.present.unwrap(),
    ];

    unique_families.sort_unstable();
    unique_families.dedup();

    let mut queue_create_infos = Vec::with_capacity(unique_families.len());
    let queue_priorities = [1.0];

    for queue_family in unique_families {
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            queue_family_index: queue_family,
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count: queue_priorities.len() as u32,
            ..Default::default()
        };

        queue_create_infos.push(queue_create_info);
    }

    let features = vk::PhysicalDeviceFeatures {
        shader_clip_distance: 1,
        ..Default::default()
    };

    let req_exts_strings = convert_to_strings(REQ_DEVICE_EXTENSIONS);
    let req_exts_cstrings = convert_to_c_strs(&req_exts_strings);
    let req_exts_cptrs = convert_to_c_ptrs(&req_exts_cstrings);

    let create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_extension_count: req_exts_cptrs.len() as u32,
        pp_enabled_extension_names: req_exts_cptrs.as_ptr(),
        p_enabled_features: &features,
        ..Default::default()
    };

    unsafe { instance.create_device(info.phys_device, &create_info, None) }
        .check_err("create device")
}

fn create_swapchain(
    phys_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &Surface,
    surface_capabilities: &vk::SurfaceCapabilitiesKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_resolution: vk::Extent2D,
    swapchain_loader: &Swapchain,
) -> vk::SwapchainKHR {
    let max_image_count = surface_capabilities.max_image_count;
    let mut image_count = surface_capabilities.min_image_count + 1;

    if image_count > max_image_count && max_image_count > 0 {
        image_count = max_image_count;
    }

    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    let present_mode = choose_swapchain_present_mode(phys_device, surface, surface_loader);

    let create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_color_space(surface_format.color_space)
        .image_format(surface_format.format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);

    unsafe { swapchain_loader.create_swapchain(&create_info, None) }.check_err("create swapchain")
}

fn choose_swapchain_present_mode(
    phys_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &Surface,
) -> vk::PresentModeKHR {
    let mut modes =
        unsafe { surface_loader.get_physical_device_surface_present_modes(phys_device, surface) }
            .check_err("get present modes");

    modes.sort_by_key(|m| present_mode_to_priority(*m));

    modes[0]
}

fn present_mode_to_priority(mode: vk::PresentModeKHR) -> u32 {
    match mode {
        vk::PresentModeKHR::IMMEDIATE => 1,
        vk::PresentModeKHR::FIFO_RELAXED => 2,
        vk::PresentModeKHR::MAILBOX => 3,
        _ => 4,
    }
}

fn create_command_pool(device: &ash::Device, queue_family_idx: u32) -> vk::CommandPool {
    let create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_idx);

    unsafe { device.create_command_pool(&create_info, None) }.check_err("create command pool")
}

fn create_command_buffers(
    device: &ash::Device,
    pool: vk::CommandPool,
    num: u32,
) -> Vec<vk::CommandBuffer> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(num)
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    unsafe { device.allocate_command_buffers(&allocate_info) }.check_err("allocate command buffers")
}

fn get_swapchain_images(
    swapchain_loader: &Swapchain,
    swapchain: vk::SwapchainKHR,
) -> Vec<vk::Image> {
    unsafe { swapchain_loader.get_swapchain_images(swapchain) }.check_err("get swapchain images")
}

fn create_image_views(
    device: &ash::Device,
    surface_format: vk::SurfaceFormatKHR,
    images: &[vk::Image],
) -> Vec<vk::ImageView> {
    images
        .iter()
        .map(|&image| {
            create_image_view(device, image, surface_format.format, vk::ImageAspectFlags::COLOR, 1)
        })
        .collect()
}

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect_mask: vk::ImageAspectFlags,
    mip_levels: u32,
) -> vk::ImageView {
    let components = vk::ComponentMapping {
        r: vk::ComponentSwizzle::IDENTITY,
        g: vk::ComponentSwizzle::IDENTITY,
        b: vk::ComponentSwizzle::IDENTITY,
        a: vk::ComponentSwizzle::IDENTITY,
    };

    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level: 0,
        level_count: mip_levels,
        base_array_layer: 0,
        layer_count: 1,
    };

    let create_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        components,
        subresource_range,
        image,
        ..Default::default()
    };

    unsafe { device.create_image_view(&create_info, None) }.check_err("create image view")
}

fn create_render_pass(device: &ash::Device, surface_format: vk::Format) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription {
        format: surface_format,
        flags: vk::AttachmentDescriptionFlags::empty(),
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
    };

    let color_attachment_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };

    let subpass = vk::SubpassDescription {
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: 1,
        p_color_attachments: &color_attachment_ref,
        ..Default::default()
    };

    let subpass_dependency = vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: vk::AccessFlags::empty(),
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: vk::DependencyFlags::empty(),
    };

    let create_info = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
        flags: vk::RenderPassCreateFlags::empty(),
        p_next: ptr::null(),
        attachment_count: 1,
        p_attachments: &color_attachment,
        subpass_count: 1,
        p_subpasses: &subpass,
        dependency_count: 1,
        p_dependencies: &subpass_dependency,
    };

    unsafe { device.create_render_pass(&create_info, None) }.check_err("create render pass")
}

fn create_pipeline_layout(device: &ash::Device) -> vk::PipelineLayout {
    let create_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
        ..Default::default()
    };

    unsafe { device.create_pipeline_layout(&create_info, None) }.check_err("create pipeline layout")
}

fn create_graphics_pipeline(
    device: &ash::Device,
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
) -> vk::Pipeline {
    let vert_compiled = include_shader!("triangle.vert");
    let frag_compiled = include_shader!("triangle.frag");

    let vert_shader_mod = create_shader_module(device, vert_compiled);
    let frag_shader_mod = create_shader_module(device, frag_compiled);

    let entrypoint_name = CString::new("main").unwrap();

    let vert_shader_stage = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage: vk::ShaderStageFlags::VERTEX,
        module: vert_shader_mod,
        p_name: entrypoint_name.as_ptr(),
        ..Default::default()
    };

    let frag_shader_stage = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage: vk::ShaderStageFlags::FRAGMENT,
        module: frag_shader_mod,
        p_name: entrypoint_name.as_ptr(),
        ..Default::default()
    };

    let shader_stages = [vert_shader_stage, frag_shader_stage];

    let binding_desc = Vertex::get_binding_desc();
    let attribute_desc = Vertex::get_attribute_desc();

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        vertex_binding_description_count: binding_desc.len() as u32,
        p_vertex_binding_descriptions: binding_desc.as_ptr(),
        vertex_attribute_description_count: attribute_desc.len() as u32,
        p_vertex_attribute_descriptions: attribute_desc.as_ptr(),
        ..Default::default()
    };

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: vk::FALSE,
        ..Default::default()
    };

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };

    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };

    let viewport_state = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        viewport_count: 1,
        p_viewports: &viewport,
        scissor_count: 1,
        p_scissors: &scissor,
        ..Default::default()
    };

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        depth_clamp_enable: vk::FALSE,
        rasterizer_discard_enable: vk::FALSE,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::CLOCKWISE,
        depth_bias_enable: vk::FALSE,
        line_width: 1.0,
        ..Default::default()
    };

    let multisample_state = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 0.0,
        p_sample_mask: ptr::null(),
        alpha_to_coverage_enable: vk::FALSE,
        alpha_to_one_enable: vk::FALSE,
        ..Default::default()
    };

    let stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
    };

    let depth_state = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        depth_test_enable: vk::FALSE,
        depth_write_enable: vk::FALSE,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: vk::FALSE,
        front: stencil_state,
        back: stencil_state,
        min_depth_bounds: 0.0,
        max_depth_bounds: 1.0,
        ..Default::default()
    };

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: 1,
        p_attachments: &color_blend_attachment,
        blend_constants: [0.0, 0.0, 0.0, 0.0],
        ..Default::default()
    };

    let create_info = [vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: &vertex_input_state,
        p_input_assembly_state: &input_assembly_state,
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state,
        p_rasterization_state: &rasterization_state,
        p_multisample_state: &multisample_state,
        p_depth_stencil_state: &depth_state,
        p_color_blend_state: &color_blend_state,
        p_dynamic_state: ptr::null(),
        layout: pipeline_layout,
        render_pass,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: -1,
    }];

    let graphics_pipelines =
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &create_info, None) };

    unsafe {
        device.destroy_shader_module(vert_shader_mod, None);
        device.destroy_shader_module(frag_shader_mod, None);
    }

    match graphics_pipelines {
        Ok(pipelines) => pipelines[0],
        Err((_pipelines, err)) => panic!("failed to create pipeline: {}", err),
    }
}

fn create_shader_module(device: &ash::Device, code: &[u8]) -> vk::ShaderModule {
    let transmuted_copy = pack_to_u32s(code);

    let create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        code_size: code.len(),
        p_code: transmuted_copy.as_ptr(),
        ..Default::default()
    };

    unsafe { device.create_shader_module(&create_info, None) }.check_err("create shader module")
}

fn pack_to_u32s(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "code length must be a multiple of 4");

    bytes
        .chunks_exact(4)
        .map(|chunk| match chunk {
            &[b0, b1, b2, b3] => u32::from_ne_bytes([b0, b1, b2, b3]),
            _ => unreachable!(),
        })
        .collect()
}

fn create_framebuffers(
    device: &ash::Device,
    image_views: &[vk::ImageView],
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
) -> Vec<vk::Framebuffer> {
    let mut framebuffers = Vec::with_capacity(image_views.len());

    for image_view in image_views {
        let create_info = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            render_pass,
            attachment_count: 1,
            p_attachments: image_view,
            width: extent.width,
            height: extent.height,
            layers: 1,
            ..Default::default()
        };

        let framebuffer = unsafe { device.create_framebuffer(&create_info, None) }
            .check_err("create framebuffer");

        framebuffers.push(framebuffer);
    }

    framebuffers
}

fn create_buffer_of_type<T: Copy>(
    device: &ash::Device,
    device_mem_properties: &vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    usage: vk::BufferUsageFlags,
    data: &[T],
) -> (vk::Buffer, vk::DeviceMemory) {
    let size_bytes: u64 = (data.len() * size_of::<T>()).try_into().unwrap();

    let (staging_buffer, staging_memory) = unsafe {
        create_buffer(
            device,
            device_mem_properties,
            size_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    };

    upload_to_buffer_memory(device, staging_memory, data);

    let (buffer, memory) = unsafe {
        create_buffer(
            device,
            device_mem_properties,
            size_bytes,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    };

    copy_buffers(device, command_pool, queue, staging_buffer, buffer, size_bytes);

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_memory, None);
    }

    (buffer, memory)
}

unsafe fn create_buffer(
    device: &ash::Device,
    device_mem_properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let create_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let buffer = device.create_buffer(&create_info, None).check_err("create buffer");

    let mem_requirements = device.get_buffer_memory_requirements(buffer);

    let memory_type =
        find_memory_type(mem_requirements.memory_type_bits, properties, device_mem_properties)
            .check_err("find appropriate memory type");

    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: mem_requirements.size,
        memory_type_index: memory_type,
        ..Default::default()
    };

    let memory = device.allocate_memory(&alloc_info, None).check_err("allocate buffer memory");

    device.bind_buffer_memory(buffer, memory, 0).check_err("bind buffer");

    (buffer, memory)
}

fn find_memory_type(
    req_type: u32,
    req_properties: vk::MemoryPropertyFlags,
    mem_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Option<u32> {
    for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
        if req_type & (1 << i) == 0 {
            continue;
        }

        if !memory_type.property_flags.contains(req_properties) {
            continue;
        }

        return Some(i as u32);
    }

    None
}

fn upload_to_buffer_memory<T: Copy>(device: &ash::Device, memory: vk::DeviceMemory, data: &[T]) {
    let size_bytes: u64 = (data.len() * size_of::<T>()).try_into().unwrap();

    let memory_range = vk::MappedMemoryRange {
        s_type: vk::StructureType::MAPPED_MEMORY_RANGE,
        memory,
        offset: 0,
        size: size_bytes,
        ..Default::default()
    };

    unsafe {
        let out_ptr = device
            .map_memory(memory, 0, size_bytes, vk::MemoryMapFlags::empty())
            .check_err("map memory");

        out_ptr.copy_from_nonoverlapping(data.as_ptr().cast::<c_void>(), data.len());

        device.flush_mapped_memory_ranges(&[memory_range]).check_err("flush mapped memory");

        device.unmap_memory(memory);
    }
}

fn copy_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: u64,
) {
    let cmd_buffer = create_command_buffers(device, command_pool, 1)[0];

    let begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    let copy_region = vk::BufferCopy {
        size,
        ..Default::default()
    };

    let submit_info = vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        command_buffer_count: 1,
        p_command_buffers: &cmd_buffer,
        ..Default::default()
    };

    unsafe {
        device.begin_command_buffer(cmd_buffer, &begin_info).check_err("begin cmd buffer");

        device.cmd_copy_buffer(cmd_buffer, src, dst, &[copy_region]);

        device.end_command_buffer(cmd_buffer).check_err("end cmd buffer");

        device.queue_submit(queue, &[submit_info], vk::Fence::null()).check_err("submit to queue");

        device.queue_wait_idle(queue).check_err("wait for queue");

        device.free_command_buffers(command_pool, &[cmd_buffer]);
    }
}

fn create_sync_objects(
    device: &ash::Device,
) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
    let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut render_finished = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut is_rendering = Vec::with_capacity(FRAMES_IN_FLIGHT);

    for _ in 0..FRAMES_IN_FLIGHT {
        image_available.push(create_semaphore(device));
        render_finished.push(create_semaphore(device));
        is_rendering.push(create_fence(device));
    }

    (image_available, render_finished, is_rendering)
}

fn create_semaphore(device: &ash::Device) -> vk::Semaphore {
    let create_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        ..Default::default()
    };

    unsafe { device.create_semaphore(&create_info, None) }.check_err("create semaphore")
}

fn create_fence(device: &ash::Device) -> vk::Fence {
    let create_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        ..Default::default()
    };

    unsafe { device.create_fence(&create_info, None) }.check_err("create fence")
}
