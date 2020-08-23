use approx::AbsDiffEq;
use ndarray::Array1;

use serde_json::Map;
use serde_json::Value;

/// Defines a way to check how well our Network has fit te data so far.
/// Used in the Network fit function to determine early stopping conditions
pub trait Metric {

    /// Returns true if the given sets of values satisfy the metric
    ///
    /// # Arguments
    ///
    /// * `o` - Actual values
    /// * `y` - Expected values
    fn call(&self, o: &Array1<f64>, y: &Array1<f64>) -> bool;
}

/// Metric that is satisfied when all values are accurate
/// to a certain number of decimal places
pub struct Accuracy {

    /// Tolerance for representing the "approximately equal to" factor
    epsilon: f64
}

impl Accuracy {
    
    /// # Arguments
    ///
    /// * `epsilon` - The delta for used to check accuracy
    /// between two values
    #[allow(dead_code)]
    pub fn new(params: &Map<String, Value>) -> Accuracy {
        let epsilon: f64 = params["epsilon"].as_f64().unwrap();

        Accuracy { epsilon: epsilon }
    }
}

impl Metric for Accuracy {
    fn call(&self, o: &Array1<f64>, y: &Array1<f64>) -> bool {
        o.abs_diff_eq(y, self.epsilon)
    }
}
