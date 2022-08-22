use ndarray::Array2;
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
    fn call(&self, o: &Array2<f64>, y: &Array2<f64>) -> bool;
}

/// Metric that is satisfied when a certain percentage
/// of all expected and actual output values are equal
pub struct Accuracy {
    /// 
    min: f64
}

impl Accuracy {
    /// 
    pub fn new(params: &Map<String, Value>) -> Self {
        let min: f64 = params["min"].as_f64().unwrap_or(1.0);
        Self { min }
    }
}

impl Metric for Accuracy {
    fn call(&self, o: &Array2<f64>, y: &Array2<f64>) -> bool {
        let equality: Vec<usize> = o.iter()
            .zip(y)
            .map(
                |a: (&f64, &f64)| 
                (a.0 == a.1) as usize
            ).collect();
        let len = equality.len() as f64;
        let sum = equality.into_iter().sum::<usize>() as f64;
        (sum / len) >= self.min
    }
}
