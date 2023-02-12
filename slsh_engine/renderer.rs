use std::default::Default;
use std::ffi::{c_char, CStr, CString};
use std::fmt::Display;
use std::mem::{size_of, transmute};
use std::ptr;
use std::str::FromStr;

use ash::extensions::khr::{Surface, Swapchain};
use ash::vk;
use glam::{Mat4, Vec3};

use crate::camera::Camera;
use crate::ui::UserInterface;
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
const REQ_VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_MESA_device_select",
    "VK_LAYER_LUNARG_monitor",
    "VK_LAYER_KHRONOS_synchronization2",
    "VK_LAYER_KHRONOS_validation",
];
const API_VER_MAJOR: u32 = 1;
const API_VER_MINOR: u32 = 0;
const API_VER_PATCH: u32 = 0;

const FRAMES_IN_FLIGHT: usize = 2;

trait CheckVkError<T> {
    fn check_err(self, action: &'static str) -> T;
}

pub struct Renderer {
    instance: ash::Instance,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_extent: vk::Extent2D,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    is_rendering: Vec<vk::Fence>,
    crosshair_push_consts: CrosshairPushConstants,
    desc_set_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_sets: Vec<vk::DescriptorSet>,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memories: Vec<vk::DeviceMemory>,
    uniform_buffers_mappings: Vec<*mut UniformBufferObject>,
    uniform_buffer_object: UniformBufferObject,
    meshes: Vec<MeshData>,
    current_frame: usize,
    current_time: f64,
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
}

#[repr(C)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

struct Mesh {
    vertices: Vec<f32>,
    indices: Vec<u16>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CrosshairPushConstants {
    proj: Mat4,
    color: Vec3,
}

struct MeshData {
    device: ash::Device,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    index_count: u32,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Renderer {
    pub unsafe fn new(app_name: &'static str, window: &Window) -> Self {
        let entry = ash::Entry::linked();
        let instance = create_instance(app_name, &entry, window);
        let surface_loader = Surface::new(&entry, &instance);
        let surface = window.create_surface(&instance);
        let phys_device_info = pick_phys_device(&instance, surface, &surface_loader);
        let phys_device = phys_device_info.phys_device;
        let device_mem_properties = instance.get_physical_device_memory_properties(phys_device);
        let device = create_logical_device(&instance, &phys_device_info);
        let gfx_queue_idx = phys_device_info.queue_family_indices.graphics.unwrap();
        let present_queue_idx = phys_device_info.queue_family_indices.present.unwrap();
        let graphics_queue = device.get_device_queue(gfx_queue_idx, 0);
        let present_queue = device.get_device_queue(present_queue_idx, 0);
        let surface_capabilities = get_surface_capabilities(phys_device, &surface_loader, surface);
        let swapchain_format = choose_swapchain_format(phys_device, &surface_loader, surface);
        let swapchain_extent = choose_swapchain_extent(window, &surface_capabilities);
        let swapchain_loader = Swapchain::new(&instance, &device);
        let swapchain = create_swapchain(
            phys_device,
            surface,
            &surface_loader,
            &surface_capabilities,
            swapchain_format,
            swapchain_extent,
            &swapchain_loader,
            &phys_device_info.queue_family_indices,
        );
        let swapchain_images = get_swapchain_images(&swapchain_loader, swapchain);
        let swapchain_image_views =
            create_image_views(&device, swapchain_format, &swapchain_images);
        let command_pool = create_command_pool(&device, gfx_queue_idx, true);
        let command_buffers =
            create_command_buffers(&device, command_pool, FRAMES_IN_FLIGHT.try_into().unwrap());
        let render_pass = create_render_pass(&device, swapchain_format.format);
        let framebuffers =
            create_framebuffers(&device, &swapchain_image_views, swapchain_extent, render_pass);
        let (image_available, render_finished, is_rendering) = create_sync_objects(&device);

        let crosshair_push_consts = CrosshairPushConstants {
            proj: Mat4::IDENTITY,
            color: Vec3::new(0.0, 1.0, 0.0),
        };

        let push_const_range = create_push_const_range::<CrosshairPushConstants>(
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        );

        let desc_set_layout = create_desc_set_layout(&device);
        let desc_pool = create_desc_pool(&device);
        let desc_sets = create_desc_sets(&device, desc_set_layout, desc_pool);

        let (uniform_buffers, uniform_buffers_memories, uniform_buffers_mappings) =
            create_uniform_buffers(&device, &device_mem_properties);

        let uniform_buffer_object = UniformBufferObject {
            model: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
        };

        fill_desc_sets(&device, &uniform_buffers, &desc_sets);

        let grid_vert_shader_compiled = include_shader!("grid.vert");
        let grid_frag_shader_compiled = include_shader!("grid.frag");

        let grid = create_grid_mesh(2.0, 32).into_mesh_data(
            device.clone(),
            &device_mem_properties,
            command_pool,
            graphics_queue,
            None,
            Some(desc_set_layout),
            grid_vert_shader_compiled,
            grid_frag_shader_compiled,
            swapchain_extent,
            render_pass,
        );

        let crosshair_vert_shader_compiled = include_shader!("crosshair.vert");
        let crosshair_frag_shader_compiled = include_shader!("crosshair.frag");

        let crosshair = create_crosshair_mesh(6.0, 2.0, window).into_mesh_data(
            device.clone(),
            &device_mem_properties,
            command_pool,
            graphics_queue,
            Some(push_const_range),
            None,
            crosshair_vert_shader_compiled,
            crosshair_frag_shader_compiled,
            swapchain_extent,
            render_pass,
        );

        let meshes = vec![grid, crosshair];

        Self {
            instance,
            surface_loader,
            surface,
            device,
            graphics_queue,
            present_queue,
            swapchain_extent,
            swapchain_loader,
            swapchain,
            swapchain_image_views,
            command_pool,
            command_buffers,
            render_pass,
            framebuffers,
            image_available,
            render_finished,
            is_rendering,
            crosshair_push_consts,
            desc_set_layout,
            desc_pool,
            desc_sets,
            uniform_buffers,
            uniform_buffers_memories,
            uniform_buffers_mappings,
            uniform_buffer_object,
            meshes,
            current_frame: 0,
            current_time: 0.0,
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
                extent: self.swapchain_extent,
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

            let push_const_stage = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;

            // unfortunately a copy, because can't find good transmute
            let push_consts_bytes: [u8; 80] = transmute(self.crosshair_push_consts);

            self.meshes[0].record_draw_commands(
                cmd_buffer,
                None,
                Some(self.desc_sets[self.current_frame]),
            );
            self.meshes[1].record_draw_commands(
                cmd_buffer,
                Some((push_const_stage, &push_consts_bytes)),
                None,
            );

            self.device.cmd_end_render_pass(cmd_buffer);

            self.device.end_command_buffer(cmd_buffer).check_err("end command buffer recording");
        }
    }

    pub fn present(&mut self) {
        let command_buffer = self.command_buffers[self.current_frame];
        let image_index = self.begin_frame();

        self.record_commands_to_buffer(command_buffer, self.framebuffers[image_index as usize]);

        self.end_frame(image_index);
    }

    fn begin_frame(&mut self) -> u32 {
        let timeout = u64::MAX;

        let image_available = self.image_available[self.current_frame];
        let is_rendering = self.is_rendering[self.current_frame];

        unsafe {
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
        }
    }

    fn end_frame(&mut self, image_index: u32) {
        let command_buffer = self.command_buffers[self.current_frame];
        let image_available = self.image_available[self.current_frame];
        let render_finished = self.render_finished[self.current_frame];
        let is_rendering = self.is_rendering[self.current_frame];

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

    pub fn update(&mut self, _dt: f64, t: f64) {
        self.current_time = t;
    }

    pub fn update_ui_push_consts(&mut self, ui: &mut UserInterface) {
        self.crosshair_push_consts.proj = *ui.proj();
    }

    pub fn update_ubo(&mut self, camera: &mut Camera) {
        self.uniform_buffer_object.view = *camera.view();
        self.uniform_buffer_object.proj = *camera.proj();

        unsafe {
            self.uniform_buffers_mappings[self.current_frame]
                .copy_from_nonoverlapping(&self.uniform_buffer_object, 1);
        }
    }

    unsafe fn cleanup_swapchain(&self) {
        self.device.device_wait_idle().unwrap();

        for fb in &self.framebuffers {
            self.device.destroy_framebuffer(*fb, None);
        }

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

            for buf in &self.uniform_buffers {
                self.device.destroy_buffer(*buf, None);
            }

            for mem in &self.uniform_buffers_memories {
                self.device.free_memory(*mem, None);
            }

            self.meshes.drain(..);

            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_descriptor_set_layout(self.desc_set_layout, None);

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Mesh {
    fn into_mesh_data(
        self,
        device: ash::Device,
        device_mem_properties: &vk::PhysicalDeviceMemoryProperties,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        push_const_range: Option<vk::PushConstantRange>,
        desc_set_layout: Option<vk::DescriptorSetLayout>,
        vert_shader_compiled: &[u8],
        frag_shader_compiled: &[u8],
        swapchain_extent: vk::Extent2D,
        render_pass: vk::RenderPass,
    ) -> MeshData {
        let (vertex_buffer, vertex_buffer_memory) = create_buffer_of_type(
            &device,
            device_mem_properties,
            command_pool,
            graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &self.vertices,
        );

        let (index_buffer, index_buffer_memory) = create_buffer_of_type(
            &device,
            device_mem_properties,
            command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &self.indices,
        );

        let index_count = self.indices.len().try_into().unwrap();

        let pipeline_layout =
            create_pipeline_layout(&device, push_const_range.as_ref(), desc_set_layout.as_ref());
        let pipeline = create_graphics_pipeline(
            &device,
            vert_shader_compiled,
            frag_shader_compiled,
            swapchain_extent,
            render_pass,
            pipeline_layout,
        );

        MeshData {
            device,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count,
            pipeline_layout,
            pipeline,
        }
    }
}

impl MeshData {
    unsafe fn record_draw_commands(
        &self,
        cmd_buffer: vk::CommandBuffer,
        push_consts: Option<(vk::ShaderStageFlags, &[u8])>,
        desc_sets: Option<vk::DescriptorSet>,
    ) {
        self.device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

        self.device.cmd_bind_vertex_buffers(cmd_buffer, 0, &[self.vertex_buffer], &[0]);

        self.device.cmd_bind_index_buffer(cmd_buffer, self.index_buffer, 0, vk::IndexType::UINT16);

        if let Some((push_const_stage_flags, push_const_bytes)) = push_consts {
            self.device.cmd_push_constants(
                cmd_buffer,
                self.pipeline_layout,
                push_const_stage_flags,
                0,
                push_const_bytes,
            );
        }

        if let Some(set) = desc_sets {
            self.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
        }

        self.device.cmd_draw_indexed(cmd_buffer, self.index_count, 1, 0, 0, 0);
    }
}

impl Drop for MeshData {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
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

    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_application_name: app_cstr.as_ptr(),
        application_version: 0,
        p_engine_name: engine_name.as_ptr(),
        engine_version,
        api_version,
        ..Default::default()
    };

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

    let flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_application_info: &app_info,
        enabled_layer_count: req_layers_cptrs.len() as u32,
        pp_enabled_layer_names: req_layers_cptrs.as_ptr(),
        enabled_extension_count: req_exts_cptrs.len() as u32,
        pp_enabled_extension_names: req_exts_cptrs.as_ptr(),
        flags,
        ..Default::default()
    };

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

        if supports_required_queues && supports_required_extensions(&extensions) {
            let info = PhysDeviceInfo {
                phys_device,
                properties,
                queue_family_indices,
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

    for format in &formats {
        if format.format == vk::Format::B8G8R8A8_UNORM
            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return *format;
        }
    }

    formats[0]
}

unsafe fn get_surface_capabilities(
    phys_device: vk::PhysicalDevice,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
) -> vk::SurfaceCapabilitiesKHR {
    surface_loader
        .get_physical_device_surface_capabilities(phys_device, surface)
        .check_err("get surface capabilities")
}

fn choose_swapchain_extent(
    window: &Window,
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }

    let win_width = window.width();
    let win_height = window.height();

    let min = capabilities.min_image_extent;
    let max = capabilities.max_image_extent;

    vk::Extent2D {
        width: win_width.clamp(min.width, max.width),
        height: win_height.clamp(min.height, max.height),
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

    let req_layers_owned = convert_to_strings(REQ_VALIDATION_LAYERS);
    let req_layers_cstrs = convert_to_c_strs(&req_layers_owned);
    let req_layers_cptrs = convert_to_c_ptrs(&req_layers_cstrs);

    let req_exts_strings = convert_to_strings(REQ_DEVICE_EXTENSIONS);
    let req_exts_cstrings = convert_to_c_strs(&req_exts_strings);
    let req_exts_cptrs = convert_to_c_ptrs(&req_exts_cstrings);

    let create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: req_layers_cptrs.len() as u32,
        pp_enabled_layer_names: req_layers_cptrs.as_ptr(),
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
    swapchain_format: vk::SurfaceFormatKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_loader: &Swapchain,
    queue_family_indices: &QueueFamilyIndices,
) -> vk::SwapchainKHR {
    let mut image_count = surface_capabilities.min_image_count + 1;
    let max_image_count = surface_capabilities.max_image_count;

    if image_count > max_image_count && max_image_count != 0 {
        image_count = max_image_count;
    }

    let present_mode = choose_swapchain_present_mode(phys_device, surface, surface_loader);

    let gfx_queue_idx = queue_family_indices.graphics.unwrap();
    let present_queue_idx = queue_family_indices.present.unwrap();

    let (image_sharing_mode, queue_family_index_count, queue_family_indices) =
        if gfx_queue_idx == present_queue_idx {
            (vk::SharingMode::EXCLUSIVE, 0, vec![])
        } else {
            (vk::SharingMode::CONCURRENT, 2, vec![gfx_queue_idx, present_queue_idx])
        };

    let create_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        surface,
        min_image_count: image_count,
        image_color_space: swapchain_format.color_space,
        image_format: swapchain_format.format,
        image_extent: swapchain_extent,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode,
        p_queue_family_indices: queue_family_indices.as_ptr(),
        queue_family_index_count,
        pre_transform: surface_capabilities.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        image_array_layers: 1,
        ..Default::default()
    };

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

fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
    reset: bool,
) -> vk::CommandPool {
    let flags = if reset {
        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
    } else {
        vk::CommandPoolCreateFlags::empty()
    };

    let create_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        flags,
        queue_family_index,
        ..Default::default()
    };

    unsafe { device.create_command_pool(&create_info, None) }.check_err("create command pool")
}

fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    num: u32,
) -> Vec<vk::CommandBuffer> {
    let allocate_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: num,
        ..Default::default()
    };

    unsafe { device.allocate_command_buffers(&allocate_info) }.check_err("allocate command buffers")
}

unsafe fn get_swapchain_images(
    swapchain_loader: &Swapchain,
    swapchain: vk::SwapchainKHR,
) -> Vec<vk::Image> {
    swapchain_loader.get_swapchain_images(swapchain).check_err("get swapchain images")
}

fn create_image_views(
    device: &ash::Device,
    swapchain_format: vk::SurfaceFormatKHR,
    images: &[vk::Image],
) -> Vec<vk::ImageView> {
    images
        .iter()
        .map(|&image| {
            create_image_view(
                device,
                image,
                swapchain_format.format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
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

fn create_render_pass(device: &ash::Device, swapchain_format: vk::Format) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format: swapchain_format,
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
        attachment_count: 1,
        p_attachments: &color_attachment,
        subpass_count: 1,
        p_subpasses: &subpass,
        dependency_count: 1,
        p_dependencies: &subpass_dependency,
        ..Default::default()
    };

    unsafe { device.create_render_pass(&create_info, None) }.check_err("create render pass")
}

fn create_pipeline_layout(
    device: &ash::Device,
    push_const_range: Option<&vk::PushConstantRange>,
    desc_set_layout: Option<&vk::DescriptorSetLayout>,
) -> vk::PipelineLayout {
    let (push_constant_range_count, p_push_constant_ranges) = match push_const_range {
        Some(range) => (1, range as *const vk::PushConstantRange),
        None => (0, ptr::null()),
    };

    let (set_layout_count, p_set_layouts) = match desc_set_layout {
        Some(set_layout) => (1, set_layout as *const vk::DescriptorSetLayout),
        None => (0, ptr::null()),
    };

    let create_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        push_constant_range_count,
        p_push_constant_ranges,
        set_layout_count,
        p_set_layouts,
        ..Default::default()
    };

    unsafe { device.create_pipeline_layout(&create_info, None) }.check_err("create pipeline layout")
}

fn create_desc_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let binding = vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::VERTEX,
        p_immutable_samplers: ptr::null(),
    };

    let create_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        binding_count: 1,
        p_bindings: &binding,
        ..Default::default()
    };

    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .check_err("create descriptor set layout")
}

fn create_graphics_pipeline(
    device: &ash::Device,
    vert_shader_compiled: &[u8],
    frag_shader_compiled: &[u8],
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
) -> vk::Pipeline {
    let vert_shader_mod = create_shader_module(device, vert_shader_compiled);
    let frag_shader_mod = create_shader_module(device, frag_shader_compiled);

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

    let size_f32 = size_of::<f32>() as u32;

    let binding_desc = vk::VertexInputBindingDescription {
        binding: 0,
        stride: size_f32 * 2,
        input_rate: vk::VertexInputRate::VERTEX,
    };

    let attribute_desc = vk::VertexInputAttributeDescription {
        binding: 0,
        location: 0,
        format: vk::Format::R32G32_SFLOAT,
        offset: 0,
    };

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        vertex_binding_description_count: 1,
        p_vertex_binding_descriptions: &binding_desc,
        vertex_attribute_description_count: 1,
        p_vertex_attribute_descriptions: &attribute_desc,
        ..Default::default()
    };

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        topology: vk::PrimitiveTopology::LINE_LIST,
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
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
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

    let memory_type_index =
        find_memory_type(mem_requirements.memory_type_bits, properties, device_mem_properties)
            .check_err("find appropriate memory type");

    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: mem_requirements.size,
        memory_type_index,
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

        return Some(i.try_into().unwrap());
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
            .check_err("map memory")
            .cast::<T>();

        out_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

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

fn create_uniform_buffers(
    device: &ash::Device,
    device_mem_properties: &vk::PhysicalDeviceMemoryProperties,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut UniformBufferObject>) {
    let mut uniform_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut uniform_buffers_memories = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut uniform_buffers_mappings = Vec::with_capacity(FRAMES_IN_FLIGHT);

    let buf_size = size_of::<UniformBufferObject>() as u64;

    for _ in 0..FRAMES_IN_FLIGHT {
        unsafe {
            let (buffer, memory) = create_buffer(
                device,
                device_mem_properties,
                buf_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let mapping = device
                .map_memory(memory, 0, buf_size, vk::MemoryMapFlags::empty())
                .check_err("map memory")
                .cast::<UniformBufferObject>();

            uniform_buffers.push(buffer);
            uniform_buffers_memories.push(memory);
            uniform_buffers_mappings.push(mapping);
        }
    }

    (uniform_buffers, uniform_buffers_memories, uniform_buffers_mappings)
}

fn create_desc_pool(device: &ash::Device) -> vk::DescriptorPool {
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: FRAMES_IN_FLIGHT as u32,
    };

    let create_info = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        max_sets: FRAMES_IN_FLIGHT as u32,
        pool_size_count: 1,
        p_pool_sizes: &pool_size,
        ..Default::default()
    };

    unsafe { device.create_descriptor_pool(&create_info, None) }.check_err("create descriptor pool")
}

fn create_desc_sets(
    device: &ash::Device,
    desc_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
) -> Vec<vk::DescriptorSet> {
    let mut layouts = Vec::with_capacity(FRAMES_IN_FLIGHT);
    layouts.resize(FRAMES_IN_FLIGHT, desc_set_layout);

    let alloc_info = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptor_pool,
        descriptor_set_count: FRAMES_IN_FLIGHT as u32,
        p_set_layouts: layouts.as_ptr(),
        ..Default::default()
    };

    unsafe { device.allocate_descriptor_sets(&alloc_info) }.check_err("allocate descriptor sets")
}

fn create_push_const_range<T>(stage_flags: vk::ShaderStageFlags) -> vk::PushConstantRange {
    vk::PushConstantRange {
        stage_flags,
        offset: 0,
        size: size_of::<T>().try_into().unwrap(),
    }
}

fn fill_desc_sets(
    device: &ash::Device,
    uniform_buffers: &[vk::Buffer],
    desc_sets: &[vk::DescriptorSet],
) {
    for i in 0..FRAMES_IN_FLIGHT {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer: uniform_buffers[i],
            offset: 0,
            range: size_of::<UniformBufferObject>() as u64,
        };

        let desc_write = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: desc_sets[i],
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info: &buffer_info,
            ..Default::default()
        };

        unsafe {
            device.update_descriptor_sets(&[desc_write], &[]);
        }
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
        is_rendering.push(create_fence(device, true));
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

fn create_fence(device: &ash::Device, signaled: bool) -> vk::Fence {
    let flags = if signaled {
        vk::FenceCreateFlags::SIGNALED
    } else {
        vk::FenceCreateFlags::empty()
    };

    let create_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags,
        ..Default::default()
    };

    unsafe { device.create_fence(&create_info, None) }.check_err("create fence")
}

fn create_grid_mesh(res: f32, cells: usize) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let min = -(res * cells as f32);
    let max = res * cells as f32;

    let mut x_off = -(res * cells as f32);
    let mut y_off = -(res * cells as f32);

    let mut idx = 0;

    for _ in 0..cells * 2 + 1 {
        vertices.push(min);
        vertices.push(y_off);
        vertices.push(max);
        vertices.push(y_off);

        indices.push(idx);
        indices.push(idx + 1);

        idx += 2;
        y_off += res;

        vertices.push(x_off);
        vertices.push(min);
        vertices.push(x_off);
        vertices.push(max);

        indices.push(idx);
        indices.push(idx + 1);

        idx += 2;
        x_off += res;
    }

    Mesh { vertices, indices }
}

fn create_crosshair_mesh(length: f32, thickness: f32, window: &Window) -> Mesh {
    // Center X and Y
    let cx = window.width() as f32 / 2.0;
    let cy = window.height() as f32 / 2.0;

    let near = thickness;
    let far = near + length;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    vertices.push(cx);
    vertices.push(cy + near + 0.5);
    vertices.push(cx);
    vertices.push(cy + far + 0.5);

    vertices.push(cx + near + 0.5);
    vertices.push(cy);
    vertices.push(cx + far + 0.5);
    vertices.push(cy);

    vertices.push(cx);
    vertices.push(cy - near - 0.5);
    vertices.push(cx);
    vertices.push(cy - far - 0.5);

    vertices.push(cx - near - 0.5);
    vertices.push(cy);
    vertices.push(cx - far - 0.5);
    vertices.push(cy);

    for i in 0..16 {
        indices.push(i);
    }

    Mesh { vertices, indices }
}
