use crate::nn::activations::ActivationFn;
use crate::nn::costs::Cost;
use crate::nn::optimizers::Optimizer;

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use rand::seq::SliceRandom;
use rand::thread_rng;

struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    inputs: Array1<f64>,
    activations: Array1<f64>,
    delta: Array1<f64>,
    neurons: usize,
    attached_layer: Option<Box<Layer>>,
    activation_fn: Box<dyn ActivationFn>,
}

impl Layer {
    fn new(neurons: usize, inputs: usize, activation_fn: Box<dyn ActivationFn>) -> Layer {
        Layer {
            weights: Array::random((neurons, inputs), Uniform::new(0., 1.)),
            biases: Array::random(neurons, Uniform::new(0., 1.)),
            inputs: Array::zeros(inputs),
            activations: Array::zeros(neurons),
            delta: Array::zeros(neurons),
            neurons: neurons,
            attached_layer: None,
            activation_fn: activation_fn,
        }
    }

    fn feed_forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let activations: Array1<f64> = self.weights.dot(inputs) + &self.biases;

        self.inputs.assign(inputs);
        self.activations.assign(&activations);

        self.activation_fn.call(&activations)
    }

    fn back_prop(&mut self, actual: &Array1<f64>, target: &Array1<f64>) {
        let prev_delta: Array1<f64>;

        match &self.attached_layer {
            Some(layer) => prev_delta = layer.weights.t().dot(&layer.delta),
            _ => prev_delta = actual - target,
        };

        let delta: Array1<f64> = self.activation_fn.prime(&self.activations) * &prev_delta;
        self.delta.assign(&delta);
    }

    fn update(&mut self, index: usize, mut optimizer: Box<dyn Optimizer>) {
        let delta: Array2<f64> = self.delta.clone().insert_axis(Axis(1));
        let inputs: Array2<f64> = self.inputs.clone().insert_axis(Axis(0));

        let gradient: Array2<f64> = delta.dot(&inputs);

        let weights: Array2<f64> = -optimizer.delta(index, gradient) + &self.weights;
        let biases: Array1<f64> = optimizer.learning_rate() * &self.delta;

        self.weights.assign(&weights);
        self.biases.assign(&biases);
    }
}

pub struct Network {
    layers: Vec<Layer>,
    optimizer: Box<dyn Optimizer>,
    cost: Box<dyn Cost>,
}

impl Network {
    pub fn new(optimizer: Box<dyn Optimizer>, cost: Box<dyn Cost>) -> Network {
        Network {
            layers: vec![],
            optimizer: optimizer,
            cost: cost,
        }
    }

    fn add_input_layer(
        &mut self,
        neurons: usize,
        inputs: usize,
        activation_fn: Box<dyn ActivationFn>,
    ) {
        self.layers.push(Layer::new(neurons, inputs, activation_fn));
    }

    fn add_hidden_layer(&mut self, neurons: usize, activation_fn: Box<dyn ActivationFn>) {
        let mut prev_layer = self.layers.last_mut().unwrap();
        let layer = Layer::new(neurons, prev_layer.neurons, activation_fn);

        let boxed = Box::new(layer);
        prev_layer.attached_layer = Some(boxed);
        // self.layers.push(boxed);
    }

    pub fn add_layer(
        &mut self,
        neurons: usize,
        inputs: Option<usize>,
        activation_fn: Box<dyn ActivationFn>,
    ) {
        match inputs {
            Some(inputs) => {
                self.add_input_layer(neurons, inputs, activation_fn);
            }
            _ => {
                self.add_hidden_layer(neurons, activation_fn);
            }
        }
    }

    pub fn fit(&self, inputs: &Vec<Array1<f64>>, outputs: &Vec<Array1<f64>>, epochs: usize) {
        let mut samples: Vec<usize> = (0..inputs.len()).collect();

        for _ in 0..epochs {
            // TODO early stop condition
            samples.shuffle(&mut thread_rng());

            for sample in samples {
                let network_output = self.predict(&inputs[sample]);

                self.layers.reverse();

                for layer in self.layers {
                    layer.back_prop(&network_output, &outputs[sample]);
                }

                self.layers.reverse();

                for (i, layer) in self.layers.iter().enumerate() {
                    layer.update(i, self.optimizer);
                }
            }
        }
        // TODO return errors
    }

    pub fn predict(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let mut output: Array1<f64> = inputs.to_owned(); // clone();

        for layer in self.layers.iter() {
            let next_output: Array1<f64> = layer.feed_forward(&output);
            output.assign(&next_output);
        }

        output
    }
}
