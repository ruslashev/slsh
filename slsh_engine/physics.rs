use glam::Vec3;

pub struct Entity {
    position: Vec3,
    velocity: Vec3,
    mass: f32,
    eye_height: f32,
}

impl Entity {
    pub fn new(pos_x: f32, pos_y: f32, pos_z: f32) -> Self {
        Self {
            position: Vec3::new(pos_x, pos_y, pos_z),
            velocity: Vec3::new(0.0, 0.0, 0.0),
            mass: 1.0,
            eye_height: 3.0,
        }
    }

    pub fn update(&mut self, dt: f64, _t: f64) {
        let dt = dt as f32;

        let acceleration = self.acceleration();

        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;

        self.detect_collisions();
    }

    pub fn eye_position(&self) -> Vec3 {
        let mut eye = self.position;
        eye.y += self.eye_height;
        eye
    }

    fn acceleration(&self) -> Vec3 {
        let gravity = Vec3::new(0.0, -9.8, 0.0);

        let acceleration = gravity / self.mass;

        acceleration
    }

    fn detect_collisions(&mut self) {
        if self.position.y >= 0.0 {
            return;
        }

        self.position.y = 0.0;
    }
}
