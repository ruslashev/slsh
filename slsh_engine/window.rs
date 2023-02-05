use std::mem::MaybeUninit;
use std::ptr;
use std::sync::mpsc::Receiver;

use ash::vk;

pub struct Window {
    glfw: glfw::Glfw,
    handle: glfw::Window,
    events: Receiver<(f64, glfw::WindowEvent)>,
}

pub enum Resolution {
    Windowed(u32, u32),
    Fullscreen,
    FullscreenWithRes(u32, u32),
}

impl Window {
    pub fn new(res: &Resolution, title: &str) -> Self {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).expect("Failed to initialize GLFW");

        glfw.window_hint(glfw::WindowHint::Visible(true));
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));

        let (handle, events) = match *res {
            Resolution::Windowed(width, height) => {
                glfw.create_window(width, height, title, glfw::WindowMode::Windowed)
            }
            Resolution::Fullscreen => glfw.with_primary_monitor(|glfw, monitor| {
                let monitor = monitor.expect("No monitors found");
                let mode = monitor.get_video_mode().expect("Failed to get video mode");
                glfw.create_window(
                    mode.width,
                    mode.height,
                    title,
                    glfw::WindowMode::FullScreen(monitor),
                )
            }),
            Resolution::FullscreenWithRes(width, height) => {
                glfw.with_primary_monitor(|glfw, monitor| {
                    let monitor = monitor.expect("No monitors found");
                    glfw.create_window(width, height, title, glfw::WindowMode::FullScreen(monitor))
                })
            }
        }
        .expect("Failed to create GLFW window");

        assert!(glfw.vulkan_supported(), "Vulkan not supported");

        Self {
            glfw,
            handle,
            events,
        }
    }

    pub fn get_required_extensions(&self) -> Vec<String> {
        self.glfw.get_required_instance_extensions().expect("Vulkan API unavaliable")
    }

    pub fn create_surface(&self, instance: &ash::Instance) -> vk::SurfaceKHR {
        let mut surface = MaybeUninit::uninit();

        self.handle
            .create_window_surface(instance.handle(), ptr::null(), surface.as_mut_ptr())
            .result()
            .expect("Failed to create window surface");

        unsafe { surface.assume_init() }
    }
}
