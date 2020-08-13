use approx::AbsDiffEq;
use ndarray::Array1;

pub trait Metric {
    fn call(&self, o: &Array1<f64>, y: &Array1<f64>) -> bool;
}

pub struct Accuracy {
    tol: f64,
}

impl Accuracy {
    pub fn new(digits: u16) -> Accuracy {
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
