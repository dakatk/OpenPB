use ndarray::{Array, Array2};

///
const BETA_1: f64 = 0.9;
///
const BETA_2: f64 = 0.999;

/// Optimizer functions that's used to determine how a Network's weights should be
/// Adjusted after each training step
pub trait Optimizer {
    /// Exposes the `learning_rate` property for all Optimizers
    fn learning_rate(&self) -> f64;
    ///
    fn next(&mut self);
    /// Returns the calculated adjustment factor for the Network's
    /// weights after a single step of training
    ///
    /// # Arguments
    ///
    /// * `index` - The numeric index of the layer currently being operated on
    /// * `gradient` - The gradient calculated between inputs
    /// and deltas for the current layer
    fn delta(&mut self, index: usize, gradient: &Array2<f64>) -> Array2<f64>;
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    /// The step size when adjusting weights for each call of gradient descent
    learning_rate: f64,
    /// Set of velocity values for use in classical momentum to ensure that
    /// steps are slowed down as they approach a minimum over time
    velocities: Vec<Array2<f64>>,
}

impl SGD {
    /// # Arguments
    ///
    /// * `momentum` - The momentum constant for adjusting the velocities using classic
    /// momentum for gradient descent. Should be less than 1, 0.9 is the ideal value
    /// * `learning_rate` - The step size when adjusting weights for each call of gradient descent
    #[allow(dead_code)]
    pub fn new(learning_rate: f64) -> SGD {
        SGD {
            learning_rate: learning_rate,
            velocities: vec![],
        }
    }
}

impl Optimizer for SGD {
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn next(&mut self) {}

    fn delta(&mut self, index: usize, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocities.len() <= index {
            self.velocities.push(Array::zeros(gradient.dim()));
        }

        let moment: Array2<f64> =
            (&self.velocities[index] * BETA_1) + (gradient * self.learning_rate);
        self.velocities[index].assign(&moment);
        moment
    }
}

pub struct Adam {
    /// 
    time_step: u16,
    /// 
    learning_rate: f64,
    /// 
    velocities: Vec<Array2<f64>>,
    /// 
    moments: Vec<Array2<f64>>,
}

impl Adam {
    ///
    #[allow(dead_code)]
    pub fn new(learning_rate: f64) -> Adam {
        Adam {
            time_step: 0,
            learning_rate: learning_rate,
            velocities: vec![],
            moments: vec![],
        }
    }
}

impl Optimizer for Adam {
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn next(&mut self) {
        self.time_step += 1;
    }

    fn delta(&mut self, index: usize, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocities.len() <= index {
            self.velocities.push(Array::zeros(gradient.dim()));
        }

        if self.moments.len() <= index {
            self.moments.push(Array::zeros(gradient.dim()));
        }

        let moment: Array2<f64> = (&self.moments[index] * BETA_1) + (gradient * (1. - BETA_1));

        let grad_squard = gradient.mapv(|el| el * el);
        let velocity: Array2<f64> =
            (&self.velocities[index] * BETA_2) + (grad_squard * (1. - BETA_2));

        self.moments[index].assign(&moment);
        self.velocities[index].assign(&velocity);
        let beta1_t = 1. - BETA_1.powi(self.time_step as i32);
        let beta2_t = 1. - BETA_2.powi(self.time_step as i32);

        let moment_bar: Array2<f64> = self.moments[index].mapv(|el| el / beta1_t);
        let velocity_bar: Array2<f64> = self.velocities[index].mapv(|el| el / beta2_t);

        let velocity_sqrt: Array2<f64> = velocity_bar.mapv(|el| f64::sqrt(el) + 1e-7);

        (moment_bar * self.learning_rate) / velocity_sqrt
    }
}
