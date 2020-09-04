use ndarray::Array1;

/// Neuron activation function used for feed forward
/// and backprop methods in Network training
pub trait ActivationFn: DynClone {
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

    // Create a clone of a boxed instance of this trait
    //fn box_clone(&self) -> Box<dyn ActivationFn> where Self : Sized;
}

pub trait DynClone {
    /// Create a clone of a boxed instance of a trait
    fn clone_box(&self) -> Box<dyn ActivationFn>;
}

impl<T> DynClone for T
where
    T: 'static + ActivationFn + Clone
{
    fn clone_box(&self) -> Box<dyn ActivationFn> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ActivationFn> {
    fn clone(&self) -> Box<dyn ActivationFn> {
        self.clone_box()
    }
}

/// Logistic Sigmoid activation function
#[derive(Clone)]
pub struct Sigmoid;

/// Mathematical definition of the Logistic Sigmoid function for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

/// Derivative of the Logistic Sigmoid function
///
/// # Arguments
///
/// * `x` - Function input value
fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1. - sigmoid(x))
}

impl ActivationFn for Sigmoid {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(sigmoid)
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
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
    if x > 0. {
        x
    } else {
        0.
    }
}

/// Derivative of the Rectified Linear Unit function
///
/// # Arguments
///
/// * `x` - Function input value
fn relu_prime(x: f64) -> f64 {
    if x > 0. {
        1.
    } else {
        0.
    }
}

impl ActivationFn for ReLU {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(relu)
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(relu_prime)
    }
}

/// Leaky Rectified Linear Unit activation function
#[derive(Clone)]
pub struct LeakyReLU;

/// Mathematical definition of the Leaky ReLU
/// function for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn leaky_relu(x: f64) -> f64 {
    if x > 0. {
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
    if x > 0. {
        1.
    } else {
        0.01
    }
}

impl ActivationFn for LeakyReLU {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(leaky_relu)
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(leaky_relu_prime)
    }
}
