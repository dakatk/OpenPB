use crate::dyn_clone;
use ndarray::{Array1, Array2, Axis};

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

fn __sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

impl ActivationFn for Sigmoid {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|x| __sigmoid(x))
    }

    fn prime(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|x| {
            let sig: f64 = __sigmoid(x);
            sig * (1.0 - sig)
        })
    }
}

/// Rectified Linear Unit activation function
#[derive(Clone)]
pub struct ReLU;

impl ActivationFn for ReLU {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn prime(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

/// "Leaky" Rectified Linear Unit activation function
#[derive(Clone)]
pub struct LeakyReLU;

impl ActivationFn for LeakyReLU {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { x } else { 0.01 * x })
    }

    fn prime(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.01 })
    }
}

/// Softmax activation function
#[derive(Clone)]
pub struct Softmax;

impl ActivationFn for Softmax {
    fn call(&self, x: &Array2<f64>) -> Array2<f64> {
        let a: Array2<f64> = x.mapv(|a| a.exp());
        let sum: Array1<f64> = a.sum_axis(Axis(0));
        a / sum
    }

    fn prime(&self, x: &Array2<f64>) -> Array2<f64> {
        let sm: Array2<f64> = self.call(x);
        let si_sj: Array2<f64> = -&sm * &sm;
        let diag: Array1<f64> = sm.diag().to_owned();
        diag + si_sj
    }
}
