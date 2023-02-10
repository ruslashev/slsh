use crate::camera::Camera;
use crate::input::InputHandler;
use crate::renderer::Renderer;
use crate::window::{Event, Key, Resolution, Window};

pub struct MainLoop {
    window: Window,
    renderer: Renderer,
    camera: Camera,
    input: InputHandler,
    running: bool,
}

impl MainLoop {
    pub fn new(res: &Resolution, app_name: &'static str) -> Self {
        let window = Window::new(res, app_name);
        let renderer = unsafe { Renderer::new(app_name, &window) };

        let aspect_ratio = window.width() as f32 / window.height() as f32;
        let camera = Camera::new(aspect_ratio);

        let (prev_mouse_x, prev_mouse_y) = window.mouse_pos();
        let input = InputHandler::new(prev_mouse_x as i32, prev_mouse_y as i32);

        Self {
            window,
            renderer,
            camera,
            input,
            running: true,
        }
    }

    pub fn run(&mut self) {
        let updates_per_second: i16 = 60;
        let dt = 1.0 / f64::from(updates_per_second);

        let mut current_time = self.window.current_time();
        let minimized = false;

        while self.running {
            if minimized {
                self.window.block_until_event();
            }

            let real_time = self.window.current_time();

            while current_time < real_time {
                current_time += dt;

                let (mouse_x, mouse_y) = self.window.mouse_pos();
                self.input.handle_mouse(mouse_x as i32, mouse_y as i32);
                self.camera.update(&self.input, dt, current_time);
                self.renderer.update(dt, current_time);
            }

            self.window.poll_events(|event| match event {
                Event::KeyPress(Key::Escape) => self.running = false,
                _ => (),
            });

            if self.window.should_close() {
                break;
            }

            self.renderer.update_ubo(&mut self.camera);
            self.renderer.present();
        }
    }
}
