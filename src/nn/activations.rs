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
    /// Clones this trait when boxed in lieu of copying
    fn box_clone(&self) -> Box<dyn ActivationFn>;
}

/// Logistic Sigmoid activation function
#[derive(Clone)]
pub struct Sigmoid;

/// Mathematical definition of Logistic Sigmoid for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

impl ActivationFn for Sigmoid {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| sigmoid(el))
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| sigmoid(el) * (1. - sigmoid(el)))
    }

    fn box_clone(&self) -> Box<dyn ActivationFn> {
        Box::new((*self).clone())
    }
}
