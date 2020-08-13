use approx::AbsDiffEq;
use ndarray::Array1;

pub trait Metric {
    fn call(&self, o: &Array1<f64>, y: &Array1<f64>) -> bool;
}

pub struct Accuracy {
    digits: u16,
}

impl Accuracy {
    pub fn new(digits: u16) -> Accuracy {
        Accuracy { digits: digits }
    }
}

impl Metric for Accuracy {
    fn call(&self, o: &Array1<f64>, y: &Array1<f64>) -> bool {
        let tol = f64::powi(10., -(self.digits as i32 + 1));
        o.abs_diff_eq(y, tol)
    }
}
