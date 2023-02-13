use crate::window::Key;

pub struct InputHandler {
    mouse_prev_x: i32,
    mouse_prev_y: i32,

    pub mouse_diff_x: i32,
    pub mouse_diff_y: i32,

    pub forward: i8,
    pub right: i8,
}

impl InputHandler {
    pub fn new(mouse_prev_x: i32, mouse_prev_y: i32) -> Self {
        Self {
            mouse_prev_x,
            mouse_prev_y,
            mouse_diff_x: 0,
            mouse_diff_y: 0,
            forward: 0,
            right: 0,
        }
    }

    pub fn handle_mouse(&mut self, x: i32, y: i32) {
        self.mouse_diff_x = x - self.mouse_prev_x;
        self.mouse_diff_y = y - self.mouse_prev_y;

        self.mouse_prev_x = x;
        self.mouse_prev_y = y;
    }

    pub fn handle_key_press(&mut self, key: Key) {
        match key {
            Key::W => self.forward = 1,
            Key::S => self.forward = -1,
            Key::D => self.right = 1,
            Key::A => self.right = -1,
            _ => (),
        }
    }

    pub fn handle_key_release(&mut self, key: Key) {
        match key {
            Key::W => {
                if self.forward == 1 {
                    self.forward = 0;
                }
            }
            Key::S => {
                if self.forward == -1 {
                    self.forward = 0;
                }
            }
            Key::D => {
                if self.right == 1 {
                    self.right = 0;
                }
            }
            Key::A => {
                if self.right == -1 {
                    self.right = 0;
                }
            }
            _ => (),
        }
    }
}
