use glam::Mat4;

pub struct UserInterface {
    win_width: u32,
    win_height: u32,
    proj: Mat4,
    proj_needs_recalc: bool,
}

impl UserInterface {
    pub fn new(win_width: u32, win_height: u32) -> Self {
        Self {
            win_width,
            win_height,
            proj: Mat4::IDENTITY,
            proj_needs_recalc: true,
        }
    }

    pub fn proj(&mut self) -> &Mat4 {
        if self.proj_needs_recalc {
            self.recalc_proj_matrix();
            self.proj_needs_recalc = false;
        }

        &self.proj
    }

    fn recalc_proj_matrix(&mut self) {
        let left = 0.0;
        let right = self.win_width as f32;
        let bottom = self.win_height as f32;
        let top = 0.0;
        let near = -1.0;
        let far = 1.0;

        self.proj = Mat4::orthographic_rh_gl(left, right, bottom, top, near, far);

        self.proj_needs_recalc = false;
    }
}
