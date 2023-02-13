use std::f32::consts::PI;

use glam::{Mat4, Vec3};

use crate::input::InputHandler;

pub struct Camera {
    fov: f32,
    near: f32,
    far: f32,

    pitch: f32,
    yaw: f32,
    roll: f32,

    pitch_min: f32,
    pitch_max: f32,

    position: Vec3,

    aspect_ratio: f32,

    proj: Mat4,
    view: Mat4,

    proj_needs_recalc: bool,
    view_needs_recalc: bool,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        Self {
            fov: 70.0_f32.to_radians(),
            near: 0.05,
            far: 100.0,
            pitch: 0.0,
            yaw: 0.0,
            roll: 0.0,
            pitch_min: -PI / 2.0 + 0.001,
            pitch_max: PI / 2.0 - 0.001,
            aspect_ratio,
            position: Vec3::new(0.0, 0.0, 0.0),
            proj: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            proj_needs_recalc: true,
            view_needs_recalc: true,
        }
    }

    pub fn proj(&mut self) -> &Mat4 {
        if self.proj_needs_recalc {
            self.recalc_proj_matrix();
            self.proj_needs_recalc = false;
        }

        &self.proj
    }

    pub fn view(&mut self) -> &Mat4 {
        if self.view_needs_recalc {
            self.recalc_view_matrix();
            self.view_needs_recalc = false;
        }

        &self.view
    }

    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn update(&mut self, input: &InputHandler, _dt: f64, _current_time: f64) {
        let sensitiviy = 2.2;
        let m_yaw = 0.022;
        let m_pitch = 0.022;
        let to_rads = PI / 180.0;

        self.pitch += input.mouse_diff_y as f32 * m_pitch * sensitiviy * to_rads;
        self.pitch = self.pitch.clamp(self.pitch_min, self.pitch_max);

        self.yaw += input.mouse_diff_x as f32 * m_yaw * sensitiviy * to_rads;

        self.view_needs_recalc = true;
    }

    fn recalc_view_matrix(&mut self) {
        self.view = Mat4::IDENTITY
            * Mat4::from_euler(glam::EulerRot::XYZ, -self.pitch, -self.yaw, -self.roll)
            * Mat4::from_translation(-self.position);

        self.view_needs_recalc = false;
    }

    fn recalc_proj_matrix(&mut self) {
        self.proj = Mat4::perspective_lh(self.fov, self.aspect_ratio, self.near, self.far);
        self.proj.y_axis.y *= -1.0;

        self.proj_needs_recalc = false;
    }
}
