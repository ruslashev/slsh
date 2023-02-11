use std::mem::MaybeUninit;
use std::ptr;
use std::sync::mpsc::Receiver;

use ash::vk;

pub struct Window {
    glfw: glfw::Glfw,
    handle: glfw::Window,
    events: Receiver<(f64, glfw::WindowEvent)>,
    width: u32,
    height: u32,
}

pub enum Resolution {
    Windowed(u32, u32),
    Fullscreen,
    FullscreenWithRes(u32, u32),
}

pub enum Event {
    KeyPress(Key),
    KeyRelease(Key),
    MouseMove(f64, f64),
}

#[repr(i32)]
pub enum Key {
    Escape = glfw::Key::Escape as i32,
    Unknown = glfw::Key::Unknown as i32,
}

impl Window {
    pub fn new(res: &Resolution, title: &str) -> Self {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).expect("Failed to initialize GLFW");

        glfw.window_hint(glfw::WindowHint::Visible(true));
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::CenterCursor(true));

        let (width, height, create_result) = match *res {
            Resolution::Windowed(width, height) => {
                let res = glfw.create_window(width, height, title, glfw::WindowMode::Windowed);

                (width, height, res)
            }
            Resolution::Fullscreen => glfw.with_primary_monitor(|glfw, monitor| {
                let monitor = monitor.expect("No monitors found");
                let mode = monitor.get_video_mode().expect("Failed to get video mode");
                let width = mode.width;
                let height = mode.height;
                let res =
                    glfw.create_window(width, height, title, glfw::WindowMode::FullScreen(monitor));

                (width, height, res)
            }),
            Resolution::FullscreenWithRes(width, height) => {
                glfw.with_primary_monitor(|glfw, monitor| {
                    let monitor = monitor.expect("No monitors found");
                    let res = glfw.create_window(
                        width,
                        height,
                        title,
                        glfw::WindowMode::FullScreen(monitor),
                    );

                    (width, height, res)
                })
            }
        };

        let (mut handle, events) = create_result.expect("Failed to create GLFW window");

        assert!(glfw.vulkan_supported(), "Vulkan not supported");

        center_window(res, &mut glfw, &mut handle);

        handle.set_key_polling(true);
        handle.set_cursor_pos_polling(true);

        handle.set_cursor_mode(glfw::CursorMode::Disabled);

        if glfw.supports_raw_motion() {
            handle.set_raw_mouse_motion(true);
        } else {
            println!("Raw mouse input not supported");
        }

        Self {
            glfw,
            handle,
            events,
            width,
            height,
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

    pub fn current_time(&self) -> f64 {
        self.glfw.get_time()
    }

    pub fn block_until_event(&mut self) {
        self.glfw.wait_events();
    }

    pub fn set_title(&mut self, title: &str) {
        self.handle.set_title(title);
    }

    pub fn mouse_pos(&self) -> (f64, f64) {
        self.handle.get_cursor_pos()
    }

    pub fn should_close(&self) -> bool {
        self.handle.should_close()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn poll_events(&mut self, mut handle_cb: impl FnMut(Event)) {
        self.glfw.poll_events();
        for (_, glfw_event) in glfw::flush_messages(&self.events) {
            match glfw_event {
                glfw::WindowEvent::Key(key, _scancode, action, _modifiers) => {
                    if action == glfw::Action::Press {
                        let event = Event::KeyPress(Key::from_glfw(key));
                        handle_cb(event);
                    }
                    if action == glfw::Action::Release {
                        let event = Event::KeyRelease(Key::from_glfw(key));
                        handle_cb(event);
                    }
                }
                glfw::WindowEvent::CursorPos(x, y) => handle_cb(Event::MouseMove(x, y)),
                _ => (),
            }
        }
    }
}

impl Key {
    fn from_glfw(key: glfw::Key) -> Self {
        match key {
            glfw::Key::Escape => Key::Escape,
            _ => Key::Unknown,
        }
    }
}

fn center_window(res: &Resolution, glfw: &mut glfw::Glfw, handle: &mut glfw::Window) {
    if let Resolution::Windowed(win_width, win_height) = *res {
        glfw.with_primary_monitor(|_, monitor| {
            let monitor = monitor.expect("No monitors found");
            let mode = monitor.get_video_mode().expect("Failed to get video mode");
            let scr_width = mode.width as i32;
            let scr_height = mode.height as i32;

            let win_width = win_width as i32;
            let win_height = win_height as i32;

            let win_x = scr_width / 2 - win_width / 2;
            let win_y = scr_height / 2 - win_height / 2;

            handle.set_pos(win_x, win_y);
        });
    }
}
