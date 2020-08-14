use approx::AbsDiffEq;
use ndarray::Array1;

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
    tol: f64,
}

impl Accuracy {
    /// # Arguments
    ///
    /// * `digits` - The number of digits after the decimal place
    /// that matter for accuracy checks. Should be no greater than 4
    pub fn new(digits: u8) -> Accuracy {
        Accuracy {
            tol: f64::powi(10., -(digits as i32)),
        }
    }
}

impl Metric for Accuracy {
    fn call(&self, o: &Array1<f64>, y: &Array1<f64>) -> bool {
        o.abs_diff_eq(y, self.tol)
    }
}