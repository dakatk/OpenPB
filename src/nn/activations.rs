use ndarray::Array1;

pub trait ActivationFn {
    fn call(&self, x: &Array1<f64>) -> Array1<f64>;
    fn prime(&self, x: &Array1<f64>) -> Array1<f64>;
    fn box_clone(&self) -> Box<dyn ActivationFn>;
}

#[derive(Clone)]
pub struct Sigmoid;

fn sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

impl ActivationFn for Sigmoid {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| sigmoid(el))
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| sigmoid(el) * (1. - sigmoid(el)))
    }

    fn box_clone(&self) -> Box<dyn ActivationFn> {
        Box::new((*self).clone())
    }
}
