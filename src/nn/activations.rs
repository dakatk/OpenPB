use ndarray::Array1;

/// Neuron activation function used for feed forward
/// and backprop methods in Network training
pub trait ActivationFn {
    /// Call the activation function with a set of inputs
    ///
    /// # Arguments
    ///
    /// * `x` - Row vector of input values
    fn call(&self, x: &Array1<f64>) -> Array1<f64>;
    /// First derivative of the activation function
    ///
    /// # Arguments
    ///
    /// * `x` - Row vector of input values
    fn prime(&self, x: &Array1<f64>) -> Array1<f64>;
    /// Create a clone of a boxed instance of this trait
    fn box_clone(&self) -> Box<dyn ActivationFn>;
}

/// Logistic Sigmoid activation function
#[derive(Clone)]
pub struct Sigmoid;

/// Rectified Linear Unit activation function
#[derive(Clone)]
pub struct ReLu;

/// Mathematical definition of Logistic Sigmoid for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

///
fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1. - sigmoid(x))
}

///
fn relu(x: f64) -> f64 {
    if x <= 0. {
        x
    } else {
        0.
    }
}

///
fn relu_prime(x: f64) -> f64 {
    if x <= 0. {
        1.
    } else {
        0.
    }
}

impl ActivationFn for Sigmoid {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| sigmoid(el))
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| sigmoid_prime(el))
    }

    fn box_clone(&self) -> Box<dyn ActivationFn> {
        Box::new((*self).clone())
    }
}

impl ActivationFn for ReLu {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| relu(el))
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| relu_prime(el))
    }

    fn box_clone(&self) -> Box<dyn ActivationFn> {
        Box::new((*self).clone())
    }
}
