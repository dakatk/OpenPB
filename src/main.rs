extern crate ndarray;
extern crate ndarray_rand;

use ndarray::{Array, Array1, Array2};

use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

trait ActivationFn {
    fn call(&self, x: Array1<f64>) -> Array1<f64>;
    fn prime(&self, x: Array1<f64>) -> Array1<f64>;
}

struct Sigmoid;

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

trait Optimizer {
    fn learning_rate(&self) -> f64;
    fn delta(&self, index: usize, gradient: Array2<f64>) -> Array2<f64>;
}

struct SGD {
    momentum: f64,
    learning_rate: f64,
    velocities: Vec<Array2<f64>>,
}

impl SGD {
    pub fn new(momentum: f64, learning_rate: f64) -> SGD {
        SGD {
            momentum: momentum,
            learning_rate: learning_rate,
            velocities: vec![],
        }
    }
}

impl Optimizer for SGD {
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn delta(&self, index: usize, gradient: Array2<f64>) -> Array2<f64> {
        if self.velocities.len() <= index {
            self.velocities.push(Array::zeros(gradient.dim()));
        };

        let moment: Array2<f64> =
            (self.velocities[index] * self.momentum) + (gradient * self.learning_rate);

        self.velocities[index].assign(&moment);
        moment
    }
}

trait Cost {
    fn prime(&self, o: Array2<f64>, y: Array2<f64>) -> Array2<f64>;
}

struct MSE;

impl Cost for MSE {
    fn prime(&self, o: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        o - y
    }
}

struct Layer<'a> {
    weights: Array2<f64>,
    biases: Array1<f64>,
    inputs: Array1<f64>,
    activations: Array1<f64>,
    delta: Array1<f64>,
    neurons: usize,
    attached_layer: Option<&'a Layer<'a>>,
    activation_fn: Box<dyn ActivationFn>,
}

impl Layer<'_> {
    pub fn new(
        neurons: usize,
        inputs: usize,
        activation_fn: Box<dyn ActivationFn>,
    ) -> Layer<'static> {
        Layer {
            weights: Array::random((neurons, inputs), Uniform::new(0., 1.)),
            biases: Array::random(neurons, Uniform::new(0., 1.)),
            inputs: Array::zeros(inputs),
            activations: Array::zeros(neurons),
            delta: Array::zeros(1),
            neurons: neurons,
            attached_layer: None,
            activation_fn: activation_fn,
        }
    }

    pub fn feed_forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let activations: Array1<f64> = self.weights.dot(inputs) + &self.biases;

        self.inputs.assign(inputs);
        self.activations.assign(&activations);

        self.activation_fn.call(activations)
    }

    pub fn back_prop(&mut self, actual: &Array1<f64>, target: &Array1<f64>) {
        match self.attached_layer {
            Some(layer) => self.delta = layer.weights.t().dot(&layer.delta),
            _ => self.delta = actual - target,
        };

        let delta: Array1<f64> = self.activation_fn.prime(self.activations) * &self.delta;
        self.delta.assign(&delta);
    }

    pub fn update(&mut self, index: usize, optimizer: Box<dyn Optimizer>) {
        let delta_size: usize = self.delta.len();
        let inputs_size: usize = self.inputs.len();

        let delta: Array2<f64> = Array2::from(vec![self.delta.iter()]);
        let inputs: Array2<f64> = Array2::from(vec![self.delta.iter()]);

        let gradient: Array2<f64> = delta.t().dot(&inputs);

        let weights: Array2<f64> = -optimizer.delta(index, gradient) + &self.weights;
        let biases: Array1<f64> = optimizer.learning_rate() * &self.delta;

        self.weights.assign(&weights);
        self.biases.assign(&biases);
    }
}

struct Network<'a> {
    layers: Vec<Layer<'a>>,
    optimizer: Box<dyn Optimizer>,
    cost: Box<dyn Cost>,
}

impl Network<'_> {
    pub fn new(optimizer: Box<dyn Optimizer>, cost: Box<dyn Cost>) {}
}

fn main() {
    println!("Hello, world!");
}
