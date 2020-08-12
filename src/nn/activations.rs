use ndarray::Array1;

pub trait ActivationFn {
    fn call(&self, x: &Array1<f64>) -> Array1<f64>;
    fn prime(&self, x: &Array1<f64>) -> Array1<f64>;
    fn box_clone(&self) -> Box<dyn ActivationFn>;
}

#[derive(Clone)]
pub struct Sigmoid;

impl Sigmoid {
    fn f(x: f64) -> f64 {
        1. / (1. + f64::exp(-x))
    }
}

impl ActivationFn for Sigmoid {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| Sigmoid::f(el))
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| Sigmoid::f(el) * (1. - Sigmoid::f(el)))
    }

    fn box_clone(&self) -> Box<dyn ActivationFn> {
        Box::new((*self).clone())
    }
}
