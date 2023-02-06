use crate::renderer::Renderer;
use crate::window::{Resolution, Window};

pub struct MainLoop {
    window: Window,
    renderer: Renderer,
}

impl MainLoop {
    pub fn new(res: Resolution, app_name: &'static str) -> Self {
        let window = Window::new(&res, app_name);
        let renderer = unsafe { Renderer::new(app_name, &window) };

        Self { window, renderer }
    }

    pub fn run(&mut self) {
        let updates_per_second: i16 = 60;
        let dt = 1.0 / f64::from(updates_per_second);

        let mut current_time = self.window.current_time_ms();
        let mut minimized = false;

        let title_update_delay = 0.03;
        let mut next_title_update_time = 0.0;

        'main_loop: while !self.window.should_close() {
            if minimized {
                self.window.block_until_event();
            }

            let real_time = self.window.current_time_ms();

            while current_time < real_time {
                current_time += dt;
                // update(dt, current_time);
            }

            // for event in self.window.poll_events() {
            //     match event {
            //         Event::KeyPress(Key::Escape) => break 'main_loop,
            //         Event::WindowResize(width, height) => {
            //             if width == 0 || height == 0 {
            //                 minimized = true;
            //                 continue 'main_loop;
            //             }

            //             minimized = false;

            //             self.renderer.handle_resize(width, height);
            //         }
            //         _ => (),
            //     }
            // }

            let draw_start = self.window.current_time_ms();

            // self.renderer.present();

            let frame_end = self.window.current_time_ms();

            if frame_end > next_title_update_time {
                next_title_update_time = frame_end + title_update_delay;

                let draw_time = frame_end - draw_start;
                let frame_time = frame_end - real_time;

                let draw_ms = draw_time * 1000.0;
                let fps = 1.0 / frame_time;

                let title = &format!("slsh | draw = {:05.2} ms, FPS = {:04.0}", draw_ms, fps);

                self.window.set_title(title);
            }
        }
    }
}
