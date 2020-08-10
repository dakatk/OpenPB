use ndarray::Array1;

pub trait Cost {
    fn prime(&self, o: &Array1<f64>, y: &Array1<f64>) -> Array1<f64>;
    fn box_clone(&self) -> Box<dyn Cost>;
}

#[derive(Clone)]
pub struct MSE;

impl Cost for MSE {
    fn prime(&self, o: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
        o - y
    }

    fn box_clone(&self) -> Box<dyn Cost> {
        Box::new((*self).clone())
    }
}
