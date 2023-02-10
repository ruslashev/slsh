use crate::renderer::Renderer;
use crate::window::{Event, Key, Resolution, Window};

pub struct MainLoop {
    window: Window,
    renderer: Renderer,
    running: bool,
}

impl MainLoop {
    pub fn new(res: &Resolution, app_name: &'static str) -> Self {
        let window = Window::new(res, app_name);
        let renderer = unsafe { Renderer::new(app_name, &window) };

        Self {
            window,
            renderer,
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
                // update(dt, current_time);
            }

            self.window.poll_events(|event| match event {
                Event::KeyPress(Key::Escape) => self.running = false,
                _ => (),
            });

            if self.window.should_close() {
                break;
            }

            self.renderer.present();
        }
    }
}
