use ndarray::Array2;

use network::Layer;

use super::network;

/// Momentum constant
const BETA_1: f64 = 0.9;

/// Secondary momentum constant
const BETA_2: f64 = 0.999;

/// Optimizer functions that's used to determine how a Network's weights should be
/// Adjusted after each training step
pub trait Optimizer {
    /// Performs gradient descent, updating weights and biases for all layers.
    /// Assumes backprop has already been done
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
    fn update(&mut self, layers: &mut Vec<Layer>) {
        if self.velocities.len() == 0 {
            for layer in layers.iter() {
                let inputs = layer.inputs.len();
                let neurons = layer.neurons;

                self.velocities.push(Array2::zeros((neurons, inputs)));
            }
        }

        for i in 0..layers.len() {
            let delta_weight: Array2<f64> = layers[i].delta.dot(&layers[i].inputs.t()) * self.learning_rate;
            let delta_bias: Array2<f64> = &layers[i].delta * self.learning_rate;

            let moment: Array2<f64> = (&self.velocities[i] * BETA_1) + (delta_weight * self.learning_rate);
            self.velocities[i].assign(&moment);

            layers[i].update(&moment, &delta_bias);
        }
    }

    /*
    fn next(&mut self, input_size: usize) -> Vec<usize> {
        let mut samples: Vec<usize> = (0..input_size).collect();

        samples.shuffle(&mut self.rng);
        samples
    }

    fn delta(&mut self, index: usize, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocities.len() <= index {
            self.velocities.push(Array::zeros(gradient.dim()));
        }

        let moment: Array2<f64> =
            (&self.velocities[index] * BETA_1) + (gradient * self.learning_rate);

        self.velocities[index].assign(&moment);

        moment
    }*/
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
        if self.velocities.len() == 0 && self.moments.len() == 0 {
            for layer in layers.iter() {
                let inputs = layer.inputs.len();
                let neurons = layer.neurons;

                self.velocities.push(Array2::zeros((neurons, inputs)));
                self.moments.push(Array2::zeros((neurons, inputs)));
            }
        }
        self.time_step += 1;

        for i in 0..layers.len() {
            let delta_weight: Array2<f64> = layers[i].delta.dot(&layers[i].inputs.t());
            let delta_bias: Array2<f64> = self.learning_rate * &layers[i].delta;

            let moment: Array2<f64> = (&self.moments[i] * BETA_1) + (&delta_weight * (1. - BETA_1));

            let velocity: Array2<f64> = {
                let grad_squard = delta_weight.mapv(|el| el * el);
                (&self.velocities[i] * BETA_2) + (grad_squard * (1. - BETA_2))
            };

            self.moments[i].assign(&moment);
            self.velocities[i].assign(&velocity);

            let moment_bar: Array2<f64> = {
                let beta1_t = 1. - BETA_1.powi(self.time_step as i32);
                self.moments[i].mapv(|el| el / beta1_t)
            };

            let velocity_sqrt: Array2<f64> = {
                let beta2_t = 1. - BETA_2.powi(self.time_step as i32);
                let velocity_bar: Array2<f64> = self.velocities[i].mapv(|el| el / beta2_t);

                velocity_bar.mapv(|el| f64::sqrt(el) + 1e-7)
            };

            let moment_adj: Array2<f64> = (moment_bar * self.learning_rate) / velocity_sqrt;

            layers[i].update(&moment_adj, &delta_bias);
        }
    }

    /*
    fn next(&mut self, input_size: usize) -> Vec<usize> {
        self.time_step += 1;
        (0..input_size).collect()
    }

    fn delta(&mut self, index: usize, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocities.len() <= index {
            self.velocities.push(Array::zeros(gradient.dim()));
        }

        if self.moments.len() <= index {
            self.moments.push(Array::zeros(gradient.dim()));
        }

        let moment: Array2<f64> = (&self.moments[index] * BETA_1) + (gradient * (1. - BETA_1));

        let velocity: Array2<f64> = {
            let grad_squard = gradient.mapv(|el| el * el);
            (&self.velocities[index] * BETA_2) + (grad_squard * (1. - BETA_2))
        };

        self.moments[index].assign(&moment);
        self.velocities[index].assign(&velocity);

        let moment_bar: Array2<f64> = {
            let beta1_t = 1. - BETA_1.powi(self.time_step as i32);
            self.moments[index].mapv(|el| el / beta1_t)
        };

        let velocity_sqrt: Array2<f64> = {
            let beta2_t = 1. - BETA_2.powi(self.time_step as i32);
            let velocity_bar: Array2<f64> = self.velocities[index].mapv(|el| el / beta2_t);

            velocity_bar.mapv(|el| f64::sqrt(el) + 1e-7)
        };

        (moment_bar * self.learning_rate) / velocity_sqrt
    }*/
}
