use glam::{Mat3, Vec3};

use crate::camera::Camera;
use crate::input::InputHandler;

const UP: Vec3 = Vec3::new(0.0, 1.0, 0.0);

const HEIGHT: f32 = 75.0;

const SPEED: f32 = 400.0;
const ACCELERATE: f32 = 10.0;
const AIR_ACCELERATE: f32 = 10.0;
const AIR_SPEED_CAP: f32 = 30.0;
const SPEED_MIN: f32 = 0.01;
const SPEED_MAX: f32 = 320.0;
const STOP_SPEED: f32 = 100.0;
const FRICTION: f32 = 8.0;
const JUMP_VEL: f32 = 15000.0;
const GRAVITY: f32 = 750.0;
const JUMP_HEIGHT: f32 = 21.0;

pub struct Entity {
    position: Vec3,
    velocity: Vec3,
    rotation: Vec3,
    mass: f32,
    eye_height: f32,
    on_ground: bool,
}

impl Entity {
    pub fn new(pos_x: f32, pos_y: f32, pos_z: f32) -> Self {
        Self {
            position: Vec3::new(pos_x, pos_y, pos_z),
            velocity: Vec3::new(0.0, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
            mass: 1.0,
            eye_height: HEIGHT,
            on_ground: false,
        }
    }

    pub fn update(&mut self, input: &InputHandler, camera: &mut Camera, dt: f64, _t: f64) {
        let dt = dt as f32;

        self.copy_orientation(camera);

        self.movement(input, dt);

        self.position += self.velocity * dt;

        self.detect_collisions();
    }

    pub fn eye_position(&self) -> Vec3 {
        let mut eye = self.position;
        eye.y += self.eye_height;
        eye
    }

    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    fn copy_orientation(&mut self, camera: &mut Camera) {
        let view = Mat3::from_rotation_y(-camera.yaw());
        let rot_x = view.x_axis[2];
        let rot_y = view.y_axis[2];
        let rot_z = view.z_axis[2];
        self.rotation = Vec3::new(rot_x, rot_y, rot_z);
    }

    fn movement(&mut self, input: &InputHandler, dt: f32) {
        self.apply_friction(dt);

        if self.on_ground {
            self.movement_ground(input, dt);
        } else {
            self.movement_air(input, dt);
        }
    }

    fn detect_collisions(&mut self) {
        if self.position.y > 0.0 {
            self.on_ground = false;
            return;
        }

        self.on_ground = true;
        self.velocity.y = 0.0;
        self.position.y = 0.0;
    }

    fn movement_ground(&mut self, input: &InputHandler, dt: f32) {
        if input.up == 1 {
            self.on_ground = false;
            self.velocity.y += JUMP_VEL * dt;
            self.movement_air(input, dt);
            return;
        }

        if input.forward == 0 && input.right == 0 {
            return;
        }

        self.accel_common(input, false, dt);
    }

    fn movement_air(&mut self, input: &InputHandler, dt: f32) {
        self.accel_common(input, true, dt);

        let gravity = Vec3::new(0.0, -GRAVITY, 0.0);

        self.velocity += gravity / self.mass * dt;
    }

    fn accel_common(&mut self, input: &InputHandler, air: bool, dt: f32) {
        let mut forward = self.rotation;
        let mut right = -forward.cross(UP).normalize();

        forward.y = 0.0;
        right.y = 0.0;

        forward = forward.normalize();
        right = right.normalize();

        let forward_move = input.forward as f32 * SPEED;
        let right_move = input.right as f32 * SPEED;

        let mut wish_vel = forward * forward_move + right * right_move;
        wish_vel.y = 0.0;

        let mut wish_dir = wish_vel;

        let mut wish_speed = wish_vel.length();

        if wish_speed < SPEED_MIN {
            // Not an issue for quake/source, but here glam's normalize() rightfully panics
            return;
        }

        if air && wish_speed > SPEED_MAX {
            wish_speed = SPEED_MAX;
        }

        wish_dir = wish_dir.normalize(); // unnecessary sqrt

        let accel = if air {
            AIR_ACCELERATE
        } else {
            ACCELERATE
        };

        self.accelerate(wish_dir, wish_speed, accel, dt);
    }

    fn accelerate(&mut self, wish_dir: Vec3, wish_speed: f32, accel: f32, dt: f32) {
        let current_speed = self.velocity.dot(wish_dir);

        let add_speed = wish_speed - current_speed;

        if add_speed <= 0.0 {
            return;
        }

        let mut accel_speed = accel * wish_speed * dt;

        if accel_speed > add_speed {
            accel_speed = add_speed;
        }

        self.velocity += wish_dir * accel_speed;
    }

    fn air_accelerate(&mut self, wish_dir: Vec3, wish_speed: f32, accel: f32, dt: f32) {
        let current_speed = self.velocity.dot(wish_dir);

        let wish_speed_capped = wish_speed.min(AIR_SPEED_CAP);

        let add_speed = wish_speed_capped - current_speed;

        if add_speed <= 0.0 {
            return;
        }

        // Not using wish_speed_capped here, as is in source engine
        let mut accel_speed = accel * wish_speed * dt;

        if accel_speed > add_speed {
            accel_speed = add_speed;
        }

        self.velocity += wish_dir * accel_speed;
    }

    fn apply_friction(&mut self, dt: f32) {
        let speed = self.velocity.length();

        if speed < SPEED_MIN {
            self.velocity = Vec3::new(0.0, 0.0, 0.0);
            return;
        }

        if !self.on_ground {
            return;
        }

        let control = speed.min(STOP_SPEED);
        let penalty = control * FRICTION * dt;
        let new_speed = (speed - penalty).max(0.0) / speed;

        self.velocity *= new_speed;
    }
}
