use crate::dyn_clone;
use ndarray::Array2;
use serde_json::{Map, Value};

/// Defines a way to check how well our Network has fit te data so far.
/// Used in the Network fit function to determine early stopping conditions
pub trait Metric: DynClone + Sync + Send {
    /// Metric name for command line output
    fn label(&self) -> &str;

    /// Metric score from comparing expected and actual values
    ///
    /// # Argumentsd
    ///
    /// * `actual` - Actual values
    /// * `expected` - Expected values
    fn value(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> f32;

    /// Returns true if the given sets of values satisfy the metric
    ///
    /// # Arguments
    ///
    /// * `actual` - Actual values
    /// * `expected` - Expected values
    fn check(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> bool;
}
dyn_clone!(Metric);

/// Metric that is satisfied when a certain percentage
/// of all expected and actual output values are equal
#[derive(Clone)]
pub struct Accuracy {
    /// Minimum passing accuracy score
    min: f32,
}

impl Accuracy {
    /// # Arguments
    ///
    /// * `params` - JSON object with initialization parameters.
    /// Allowed keys: "min"
    pub fn new(params: &Map<String, Value>) -> Self {
        let min: f64 = params["min"].as_f64().unwrap_or(1.0);
        Self { min: min as f32 }
    }
}

impl Metric for Accuracy {
    fn label(&self) -> &str {
        "Accuracy"
    }

    fn value(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> f32 {
        let equality: Vec<usize> = actual
            .iter()
            .zip(expected)
            .map(|a: (&f64, &f64)| (a.0 == a.1) as usize)
            .collect();
        let len = equality.len() as f32;
        let sum = equality.into_iter().sum::<usize>() as f32;
        sum / len
    }

    fn check(&self, actual: &Array2<f64>, expected: &Array2<f64>) -> bool {
        self.value(actual, expected) >= self.min
    }
}
