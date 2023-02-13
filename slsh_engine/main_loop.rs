use crate::camera::Camera;
use crate::input::InputHandler;
use crate::physics::Entity;
use crate::renderer::Renderer;
use crate::ui::UserInterface;
use crate::window::{Event, Key, Resolution, Window};

pub struct MainLoop {
    window: Window,
    renderer: Renderer,
    camera: Camera,
    input: InputHandler,
    ui: UserInterface,
    player: Entity,
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

        let ui = UserInterface::new(window.width(), window.height());

        let player = Entity::new(0.0, 8.0, 0.0);

        Self {
            window,
            renderer,
            camera,
            input,
            ui,
            player,
            running: true,
        }
    }

    pub fn run(&mut self) {
        let updates_per_second: i16 = 60;
        let dt = 1.0 / f64::from(updates_per_second);

        let title_update_delay = 0.1;
        let mut next_title_update_time = 0.0;

        let mut current_time = self.window.current_time();
        let minimized = false;

        while self.running {
            if minimized {
                self.window.block_until_event();
            }

            self.window.poll_events(|event| match event {
                Event::KeyPress(Key::Escape) => self.running = false,
                Event::KeyPress(key) => self.input.handle_key_press(key),
                Event::KeyRelease(key) => self.input.handle_key_release(key),
                _ => (),
            });

            let real_time = self.window.current_time();

            while current_time < real_time {
                current_time += dt;

                let (mouse_x, mouse_y) = self.window.mouse_pos();
                self.input.handle_mouse(mouse_x as i32, mouse_y as i32);
                self.player.update(&self.input, &mut self.camera, dt, current_time);
                self.camera.set_position(self.player.eye_position());
                self.camera.update(&self.input, dt, current_time);
                self.renderer.update(dt, current_time);
            }

            if self.window.should_close() {
                break;
            }

            let draw_start = self.window.current_time();

            self.renderer.update_data(&mut self.ui, &mut self.camera);
            self.renderer.present();

            let frame_end = self.window.current_time();

            if frame_end > next_title_update_time {
                next_title_update_time = frame_end + title_update_delay;

                let draw_time = frame_end - draw_start;
                let frame_time = frame_end - real_time;

                let draw_ms = draw_time * 1000.0;
                let fps = 1.0 / frame_time;

                let title = format!("slsh | draw = {:05.2} ms, FPS = {:04.0}", draw_ms, fps);

                self.window.set_title(&title);
            }
        }
    }
}
