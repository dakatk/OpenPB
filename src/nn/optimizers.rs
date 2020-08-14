use ndarray::{Array, Array2};

/// Optimizer functions that's used to determine how a Network's weights should be
/// Adjusted after each training step
pub trait Optimizer {
    /// Exposes the `learning_rate` property for all Optimizers
    fn learning_rate(&self) -> f64;
    /// Returns the calculated adjustment factor for the Network's
    /// weights after a single step of training
    ///
    /// # Arguments
    ///
    /// * `index` - The numeric index of the layer currently being operated on
    /// * `gradient` - The gradient calculated between inputs
    /// and deltas for the current layer
    fn delta(&mut self, index: usize, gradient: Array2<f64>) -> Array2<f64>;
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    /// The momentum constant for adjusting the velocities using classic
    /// momentum for gradient descent. Should be less than 1, 0.9 is the ideal value
    momentum: f64,
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
    pub fn new(momentum: f64, learning_rate: f64) -> SGD {
        SGD {
            momentum: momentum,
            learning_rate: learning_rate,
            velocities: vec![],
        }
    }
}

impl Optimizer for SGD {
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn delta(&mut self, index: usize, gradient: Array2<f64>) -> Array2<f64> {
        if self.velocities.len() <= index {
            self.velocities.push(Array::zeros(gradient.dim()));
        };

        let moment: Array2<f64> =
            (&self.velocities[index] * self.momentum) + (gradient * self.learning_rate);
        self.velocities[index].assign(&moment);
        moment
    }
}
