use std::default::Default;
use std::ffi::{c_char, CStr, CString};
use std::fmt::Display;
use std::ptr;
use std::str::FromStr;

use ash::extensions::khr::{Surface, Swapchain};
use ash::vk;

use crate::window::Window;

const REQ_DEVICE_EXTENSIONS: &[&str] = &[
    "VK_KHR_swapchain",
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    "VK_KHR_portability_subset",
];
const REQ_VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];
const API_VER_MAJOR: u32 = 1;
const API_VER_MINOR: u32 = 0;
const API_VER_PATCH: u32 = 0;

const FRAMES_IN_FLIGHT: u32 = 2;

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
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    render_pass: vk::RenderPass,
}

#[derive(Clone)]
struct PhysDeviceInfo {
    phys_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    queue_families: Vec<vk::QueueFamilyProperties>,
    extensions: Vec<vk::ExtensionProperties>,
    queue_family_idx: u32,
}

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
        let surface_format = choose_swapchain_format(phys_device, &surface_loader, surface);
        let surface_capabilities = get_surface_capabilities(phys_device, &surface_loader, surface);
        let surface_resolution = choose_swapchain_extent(window, &surface_capabilities);
        let device = create_logical_device(&instance, &phys_device_info);
        let present_queue = device.get_device_queue(phys_device_info.queue_family_idx, 0);
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
        let command_pool = create_command_pool(&device, phys_device_info.queue_family_idx);
        let command_buffers = create_command_buffers(&device, command_pool, FRAMES_IN_FLIGHT + 1);
        let render_pass = create_render_pass(&device, surface_format.format);

        Self {
            entry,
            instance,
            surface_loader,
            surface,
            surface_format,
            surface_resolution,
            phys_device,
            device,
            swapchain_loader,
            swapchain,
            swapchain_image_views,
            command_pool,
            command_buffers,
            render_pass,
        }
    }

    unsafe fn cleanup_swapchain(&self) {
        self.device.device_wait_idle().unwrap();

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

            self.cleanup_swapchain();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
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
        let queue_families = instance.get_physical_device_queue_family_properties(phys_device);
        let extensions = instance
            .enumerate_device_extension_properties(phys_device)
            .check_err("enumerate device extensions");

        let name = CStr::from_ptr(properties.device_name.as_ptr()).to_str().unwrap();
        println!("\t{}", name);

        let mut queue_family = None;

        for (i, qf) in queue_families.iter().enumerate() {
            let idx: u32 = i.try_into().unwrap();
            let graphics_support = qf.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let surface_support = surface_loader
                .get_physical_device_surface_support(phys_device, idx, surface)
                .check_err("get surface support");

            if graphics_support && surface_support {
                queue_family = Some(idx);
                break;
            }
        }

        if !supports_required_extensions(&extensions) {
            continue;
        }

        if let Some(queue_family_idx) = queue_family {
            let info = PhysDeviceInfo {
                phys_device,
                properties,
                queue_families,
                extensions,
                queue_family_idx,
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

fn create_logical_device(instance: &ash::Instance, info: &PhysDeviceInfo) -> ash::Device {
    let features = vk::PhysicalDeviceFeatures {
        shader_clip_distance: 1,
        ..Default::default()
    };
    let priorities = [1.0];

    let req_exts_strings = convert_to_strings(REQ_DEVICE_EXTENSIONS);
    let req_exts_cstrings = convert_to_c_strs(&req_exts_strings);
    let req_exts_cptrs = convert_to_c_ptrs(&req_exts_cstrings);

    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(info.queue_family_idx)
        .queue_priorities(&priorities);

    let create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&req_exts_cptrs)
        .enabled_features(&features);

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

    let present_modes =
        unsafe { surface_loader.get_physical_device_surface_present_modes(phys_device, surface) }
            .check_err("get present modes");

    let present_mode = present_modes
        .iter()
        .copied()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

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
        p_attachments: [color_attachment].as_ptr(),
        subpass_count: 1,
        p_subpasses: [subpass].as_ptr(),
        dependency_count: 1,
        p_dependencies: [subpass_dependency].as_ptr(),
    };

    unsafe { device.create_render_pass(&create_info, None) }.check_err("create render pass")
}
