pub struct InputHandler {
    mouse_prev_x: i32,
    mouse_prev_y: i32,

    pub mouse_diff_x: i32,
    pub mouse_diff_y: i32,
}

impl InputHandler {
    pub fn new(mouse_prev_x: i32, mouse_prev_y: i32) -> Self {
        Self {
            mouse_prev_x,
            mouse_prev_y,
            mouse_diff_x: 0,
            mouse_diff_y: 0,
        }
    }

    pub fn handle_mouse(&mut self, x: i32, y: i32) {
        self.mouse_diff_x = x - self.mouse_prev_x;
        self.mouse_diff_y = y - self.mouse_prev_y;

        self.mouse_prev_x = x;
        self.mouse_prev_y = y;
    }
}
