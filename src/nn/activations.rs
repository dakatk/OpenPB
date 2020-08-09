use ndarray::Array1;

pub trait ActivationFn {
    fn call(&self, x: Array1<f64>) -> Array1<f64>;
    fn prime(&self, x: Array1<f64>) -> Array1<f64>;
}

pub struct Sigmoid;

impl ActivationFn for Sigmoid {
    fn call(&self, x: Array1<f64>) -> Array1<f64> {
        x.mapv(|el| 1. / 1. + f64::exp(-el))
    }

    fn prime(&self, x: Array1<f64>) -> Array1<f64> {
        fn sigmoid(el: f64) -> f64 {
            1. / 1. + f64::exp(-el)
        }

        x.mapv(|el| sigmoid(el) * (1. - sigmoid(el)))
    }
}
