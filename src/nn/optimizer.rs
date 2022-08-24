use super::layer::Layer;
use ndarray::Array2;

/// Wrapper for updating a network with any given
/// optimization function using online training
pub fn optimize(optimizer: &mut dyn Optimizer, layers: &mut Vec<Layer>, input_rows: usize) {
    // TODO Minibatch support?
    let deltas: Vec<Array2<f64>> = layers.iter().map(|l| l.delta.clone()).collect();
    optimizer.update(layers, &deltas, input_rows);
}

/// Default momentum constant
pub const DEFAULT_GAMMA: f64 = 0.9;

/// Default secondary momentum constant
pub const DEFAULT_BETA: f64 = 0.999;

/// Optimizer functions that's used to determine how a Network's weights should be
/// Adjusted after each training step
pub trait Optimizer: DynClone + Sync + Send {
    /// Returns the calculated adjustment factor for the Network's
    /// weights after a single step of training
    ///
    /// # Arguments
    ///
    /// * `layers` - Layers of the network to apply gradient descent to
    fn update(&mut self, layers: &mut Vec<Layer>, deltas: &Vec<Array2<f64>>, input_rows: usize);
}

/// Stochastic Gradient Descent with momentum
#[derive(Clone)]
pub struct SGD {
    /// The step size when adjusting weights for each call of gradient descent
    learning_rate: f64,

    /// Momentum constant, typically set to 0.9 (`DEFAULT_GAMMA`) except
    /// in certain edge cases
    gamma: f64,

    /// Set of moment values for use in classical momentum
    moments: Vec<Array2<f64>>,
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
            moments: vec![],
        }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layers: &mut Vec<Layer>, deltas: &Vec<Array2<f64>>, input_rows: usize) {
        for (i, layer) in layers.iter_mut().enumerate() {
            // Convert activation (z) deltas from initial back-prop run
            // into weight and bias deltas
            let delta_weights: Array2<f64> = self.learning_rate * deltas[i].dot(&layer.inputs.t());
            let delta_biases: Array2<f64> = self.learning_rate * &deltas[i];

            // Create momentum vectors if they don't already exist
            if self.moments.len() <= i {
                self.moments.push(Array2::zeros(delta_weights.dim()));
            }

            // Apply momentum to weight deltas
            let moment: Array2<f64> = {
                let prev_moment: Array2<f64> = self.moments[i].clone();
                (self.gamma * prev_moment) + &delta_weights
            };

            // Apply deltas to layer
            layer.update(&moment, &delta_biases, input_rows);
            // Save momentum values for future passes
            self.moments[i].assign(&moment);
        }
    }
}

#[derive(Clone)]
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
    moments: Vec<Array2<f64>>,
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
            moments: vec![],
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, layers: &mut Vec<Layer>, deltas: &Vec<Array2<f64>>, input_rows: usize) {
        self.time_step += 1;

        for (i, layer) in layers.iter_mut().enumerate() {
            // Convert activation (z) deltas from initial back-prop run
            // into weight and bias deltas
            let delta_weights: Array2<f64> = deltas[i].dot(&layer.inputs.t());
            let delta_biases: Array2<f64> = self.learning_rate * &deltas[i];

            // Create velocity vectors if they don't already exist
            if self.velocities.len() <= i {
                self.velocities.push(Array2::zeros(delta_weights.dim()));
            }

            // Create momentum vectors if they don't already exist
            if self.moments.len() <= i {
                self.moments.push(Array2::zeros(delta_weights.dim()));
            }

            // Initial momentum calculation
            let moment: Array2<f64> =
                (&self.moments[i] * self.gamma) + (&delta_weights * (1. - self.gamma));

            // Initial velocity calculation
            let velocity: Array2<f64> = {
                let grad_squared = delta_weights.mapv(|el| el * el);
                (&self.velocities[i] * self.beta) + (grad_squared * (1. - self.beta))
            };

            // Save momentum and velocity values for future passes
            self.moments[i].assign(&moment);
            self.velocities[i].assign(&velocity);

            // Adjust momentum inversely relative to the number of training cycles
            let moment_bar: Array2<f64> = {
                let beta1_t = 1. - self.gamma.powi(self.time_step as i32);
                self.moments[i].mapv(|el| el / beta1_t)
            };

            // Adjust velocity inversely relative to the number of training cycles
            let velocity_sqrt: Array2<f64> = {
                let beta2_t = 1. - self.beta.powi(self.time_step as i32);
                let velocity_bar: Array2<f64> = self.velocities[i].mapv(|el| el / beta2_t);

                velocity_bar.mapv(|el| f64::sqrt(el) + 1e-7)
            };

            // Calculate final momentum w.r.t. velocity
            let moment_adj: Array2<f64> = (moment_bar * self.learning_rate) / velocity_sqrt;
            layer.update(&moment_adj, &delta_biases, input_rows)
        }
    }
}

pub trait DynClone {
    /// Create a clone of a boxed instance of a trait
    fn clone_box(&self) -> Box<dyn Optimizer>;
}

impl<T> DynClone for T
where
    T: 'static + Optimizer + Clone,
{
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Optimizer> {
    fn clone(&self) -> Box<dyn Optimizer> {
        self.clone_box()
    }
}
