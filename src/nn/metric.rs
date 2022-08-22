use ndarray::Array2;
use serde_json::{Map, Value};

/// Defines a way to check how well our Network has fit te data so far.
/// Used in the Network fit function to determine early stopping conditions
pub trait Metric {
    /// Metric name for command line output
    fn label(&self) -> &str;

    ///
    fn value(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> f64;

    /// Returns true if the given sets of values satisfy the metric
    ///
    /// # Arguments
    ///
    /// * `o` - Actual values
    /// * `y` - Expected values
    fn check(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> bool;
}

/// Metric that is satisfied when a certain percentage
/// of all expected and actual output values are equal
pub struct Accuracy {
    ///
    min: f64,
}

impl Accuracy {
    ///
    pub fn new(params: &Map<String, Value>) -> Self {
        let min: f64 = params["min"].as_f64().unwrap_or(1.0);
        Self { min }
    }
}

impl Metric for Accuracy {
    fn label(&self) -> &str {
        "Accuracy"
    }

    fn value(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> f64 {
        let equality: Vec<usize> = actual
            .iter()
            .zip(expected)
            .map(|a: (&f64, &f64)| (a.0 == a.1) as usize)
            .collect();
        let len = equality.len() as f64;
        let sum = equality.into_iter().sum::<usize>() as f64;
        sum / len
    }

    fn check(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> bool {
        self.value(actual, expected) >= self.min
    }
}
