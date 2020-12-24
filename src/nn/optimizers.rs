use ndarray::Array2;
use super::layer::Layer;

/// Default momentum constant
pub const DEFAULT_BETA_1: f64 = 0.9;

/// Default secondary momentum constant
pub const DEFAULT_BETA_2: f64 = 0.999;

/// Optimizer functions that's used to determine how a Network's weights should be
/// Adjusted after each training step
pub trait Optimizer {
    /// Returns the calculated adjustment factor for the Network's
    /// weights after a single step of training
    ///
    /// # Arguments
    ///
    /// * `layers` - layers of the network to apply gradient descent to
    fn update(&mut self, layers: &mut Vec<Layer>);
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    /// The step size when adjusting weights for each call of gradient descent
    learning_rate: f64,

    /// Set of velocity values for use in classical momentum to ensure that
    /// steps are slowed down as they approach a minimum over time
    velocities: Vec<Array2<f64>>
}

impl SGD {
    /// # Arguments
    ///
    /// * `learning_rate` - The step size when adjusting weights for each call of gradient descent
    #[allow(dead_code)]
    pub fn new(learning_rate: f64) -> SGD {
        SGD {
            learning_rate,
            velocities: vec![]
        }
    }
}

impl Optimizer for SGD {
    // TODO add batch support
    fn update(&mut self, layers: &mut Vec<Layer>) {

        for i in 0..layers.len() {
            let delta_weights: Array2<f64> = layers[i].delta.dot(&layers[i].inputs.t());
            let delta_biases: Array2<f64> = self.learning_rate * &layers[i].delta;

            if self.velocities.len() <= i {
                self.velocities.push(Array2::zeros(delta_weights.dim()));
            }
            
            let moment: Array2<f64> =
                (&self.velocities[i] * DEFAULT_BETA_1) + (delta_weights * self.learning_rate);

            self.velocities[i].assign(&moment);

            layers[i].update(&moment, &delta_biases);
        }
    }
}

pub struct Adam {
    /// Current step in the training process
    time_step: u16,

    /// The step size when adjusting weights for each call of gradient descent
    learning_rate: f64,

    /// Set of velocity values for use in RMS propogation
    velocities: Vec<Array2<f64>>,

    /// Set of moment values for use in classical momentum
    moments: Vec<Array2<f64>>
}

impl Adam {
    ///
    #[allow(dead_code)]
    pub fn new(learning_rate: f64) -> Adam {
        Adam {
            time_step: 0,
            learning_rate,
            velocities: vec![],
            moments: vec![]
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, layers: &mut Vec<Layer>) {
        self.time_step += 1;
        
        for i in 0..layers.len() {
            let delta_weights: Array2<f64> = layers[i].delta.dot(&layers[i].inputs.t());
            let delta_biases: Array2<f64> = self.learning_rate * &layers[i].delta;

            if self.velocities.len() <= i {
                self.velocities.push(Array2::zeros(delta_weights.dim()));
            }

            if self.moments.len() <= i {
                self.moments.push(Array2::zeros(delta_weights.dim()));
            }

            let moment: Array2<f64> = (&self.moments[i] * DEFAULT_BETA_1) + (&delta_weights * (1. - DEFAULT_BETA_1));

            let velocity: Array2<f64> = {
                let grad_squard = delta_weights.mapv(|el| el * el);
                (&self.velocities[i] * DEFAULT_BETA_2) + (grad_squard * (1. - DEFAULT_BETA_2))
            };

            self.moments[i].assign(&moment);
            self.velocities[i].assign(&velocity);

            let moment_bar: Array2<f64> = {
                let beta1_t = 1. - DEFAULT_BETA_1.powi(self.time_step as i32);
                self.moments[i].mapv(|el| el / beta1_t)
            };

            let velocity_sqrt: Array2<f64> = {
                let beta2_t = 1. - DEFAULT_BETA_2.powi(self.time_step as i32);
                let velocity_bar: Array2<f64> = self.velocities[i].mapv(|el| el / beta2_t);

                velocity_bar.mapv(|el| f64::sqrt(el) + 1e-7)
            };

            let moment_adj = (moment_bar * self.learning_rate) / velocity_sqrt;

            layers[i].update(&moment_adj, &delta_biases)
        }
    }
}