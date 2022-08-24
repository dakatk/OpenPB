use crate::dyn_clone;
use ndarray::Array2;

/// Neuron activation function used for feed forward
/// and backprop methods in Network training
pub trait ActivationFn: DynClone + Sync + Send {
    /// Call the activation function with a set of inputs
    ///
    /// # Arguments
    ///
    /// * `x` - Row vector of input values
    fn call(&self, x: &Array2<f64>) -> Array2<f64>;

    /// First derivative of the activation function
    ///
    /// # Arguments
    ///
    /// * `x` - Row vector of input values
    fn prime(&self, x: &Array2<f64>) -> Array2<f64>;
}
dyn_clone!(ActivationFn);

/// Logistic Sigmoid activation function
#[derive(Clone)]
pub struct Sigmoid;

/// Mathematical definition of the Logistic Sigmoid function for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

/// Derivative of the Logistic Sigmoid function
///
/// # Arguments
///
/// * `x` - Function input value
fn sigmoid_prime(x: f64) -> f64 {
    let sig: f64 = sigmoid(x);
    sig * (1.0 - sig)
}

impl ActivationFn for Sigmoid {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(sigmoid)
    }

    fn prime(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(sigmoid_prime)
    }
}

/// Rectified Linear Unit activation function
#[derive(Clone)]
pub struct ReLU;

/// Mathematical definition of the Rectified Linear Unit
/// function for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Derivative of the Rectified Linear Unit function
///
/// # Arguments
///
/// * `x` - Function input value
fn relu_prime(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

impl ActivationFn for ReLU {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(relu)
    }

    fn prime(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(relu_prime)
    }
}

/// "Leaky" Rectified Linear Unit activation function
#[derive(Clone)]
pub struct LeakyReLU;

/// Mathematical definition of the Leaky ReLU
/// function for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

/// Derivative of the Leaky ReLU function
///
/// # Arguments
///
/// * `x` - Function input value
fn leaky_relu_prime(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.01
    }
}

impl ActivationFn for LeakyReLU {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(leaky_relu)
    }

    fn prime(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(leaky_relu_prime)
    }
}
