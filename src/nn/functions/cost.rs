use crate::dyn_clone;
use ndarray::Array2;

/// Cost or loss function to determine the Network's error
pub trait Cost: DynClone + Sync + Send {
    /// First derivative of the cost function. Used in Network backprop
    ///
    /// # Arguments
    ///
    /// * `actual` - Actual values
    /// * `expected` - Expected values
    fn prime(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> Array2<f64>;
}
dyn_clone!(Cost);

/// Mean Squared Error loss function
#[derive(Clone)]
pub struct MSE;

impl Cost for MSE {
    fn prime(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> Array2<f64> {
        actual - expected
    }
}
