use super::layer::Layer;
use ndarray::Array2;

/// Wrapper for updating a network with any given 
/// optimization function using online training
pub fn optimize(optimizer: &mut dyn Optimizer, layers: &mut Vec<Layer>, input_rows: usize) {
    let deltas: Vec<Array2<f64>> = layers.iter().map(|l| l.delta.clone()).collect();
    optimizer.update(layers, &deltas, input_rows);
}

/// Default momentum constant
pub const DEFAULT_GAMMA: f64 = 0.9;

/// Default secondary momentum constant
pub const DEFAULT_BETA: f64 = 0.999;

/// Optimizer functions that's used to determine how a Network's weights should be
/// Adjusted after each training step
pub trait Optimizer {
    /// Returns the calculated adjustment factor for the Network's
    /// weights after a single step of training
    ///
    /// # Arguments
    ///
    /// * `layers` - Layers of the network to apply gradient descent to
    fn update(&mut self, layers: &mut Vec<Layer>, deltas: &Vec<Array2<f64>>, input_rows: usize);
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    /// The step size when adjusting weights for each call of gradient descent
    learning_rate: f64,

    /// Momentum constant, typically set to 0.9 (`DEFAULT_GAMMA`) except
    /// in certain edge cases
    gamma: f64,

    /// Set of moment values for use in classical momentum
    moments: Vec<Array2<f64>>
}

impl SGD {
    /// # Arguments
    ///
    /// * `learning_rate` - The step size when adjusting weights during gradient descent
    #[allow(dead_code)]
    pub fn new(learning_rate: f64, gamma: f64) -> SGD {
        SGD {
            learning_rate,
            gamma,
            moments: vec![]
        }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layers: &mut Vec<Layer>, deltas: &Vec<Array2<f64>>, input_rows: usize) {
        for (i, layer) in layers.iter_mut().enumerate() {
            let delta_weights: Array2<f64> = self.learning_rate * deltas[i].dot(&layer.inputs.t());
            let delta_biases: Array2<f64> = self.learning_rate * &deltas[i];

            if self.moments.len() <= i {
                self.moments.push(Array2::zeros(delta_weights.dim()));
            }

            let moment: Array2<f64> = {
                let prev_moment = self.moments[i].clone();
                self.gamma * prev_moment + &delta_weights
            };

            layer.update(&moment, &delta_biases, input_rows);
            self.moments[i] = moment;
        }
    }
}

pub struct Adam {
    /// Current step in the training process
    time_step: u16,

    /// The step size when adjusting weights during gradient descent
    learning_rate: f64,

    /// Momentum constant, typically set to 0.9 (`DEFAULT_GAMMA`) except
    /// in certain edge cases
    gamma: f64,

    /// Secondary momentum constant, typically set to 0.999 (`DEFAULT_BETA`) except
    /// in certain edge cases
    beta: f64,

    /// Set of velocity values for use in RMS propogation
    velocities: Vec<Array2<f64>>,

    /// Set of moment values for use in classical momentum
    moments: Vec<Array2<f64>>
}

impl Adam {
    /// # Arguments
    ///
    /// * `learning_rate` - The step size when adjusting weights during gradient descent
    #[allow(dead_code)]
    pub fn new(learning_rate: f64, gamma: f64, beta: f64) -> Adam {
        Adam {
            time_step: 0,
            learning_rate,
            gamma,
            beta,
            velocities: vec![],
            moments: vec![]
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, layers: &mut Vec<Layer>, deltas: &Vec<Array2<f64>>, input_rows: usize) {
        self.time_step += 1;

        for (i, layer) in layers.iter_mut().enumerate() {
            let delta_weights: Array2<f64> = deltas[i].dot(&layer.inputs.t());
            let delta_biases: Array2<f64> = self.learning_rate * &deltas[i];

            if self.velocities.len() <= i {
                self.velocities.push(Array2::zeros(delta_weights.dim()));
            }

            if self.moments.len() <= i {
                self.moments.push(Array2::zeros(delta_weights.dim()));
            }

            let moment: Array2<f64> =
                (&self.moments[i] * self.gamma) + (&delta_weights * (1. - self.gamma));

            let velocity: Array2<f64> = {
                let grad_squard = delta_weights.mapv(|el| el * el);
                (&self.velocities[i] * self.beta) + (grad_squard * (1. - self.beta))
            };

            self.moments[i].assign(&moment);
            self.velocities[i].assign(&velocity);

            let moment_bar: Array2<f64> = {
                let beta1_t = 1. - self.gamma.powi(self.time_step as i32);
                self.moments[i].mapv(|el| el / beta1_t)
            };

            let velocity_sqrt: Array2<f64> = {
                let beta2_t = 1. - self.beta.powi(self.time_step as i32);
                let velocity_bar: Array2<f64> = self.velocities[i].mapv(|el| el / beta2_t);

                velocity_bar.mapv(|el| f64::sqrt(el) + 1e-7)
            };

            let moment_adj: Array2<f64> = (moment_bar * self.learning_rate) / velocity_sqrt;
            layer.update(&moment_adj, &delta_biases, input_rows)
        }
    }
}
