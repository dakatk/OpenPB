use ndarray::Array2;

pub trait Cost {
    fn prime(&self, o: &Array2<f64>, y: &Array2<f64>) -> Array2<f64>;
}

pub struct MSE;

impl Cost for MSE {
    fn prime(&self, o: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        o - y
    }
}
