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
            activation_fn: activation_fn,
        }
    }

    fn feed_forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let activations: Array1<f64> = self.weights.dot(inputs) + &self.biases;

        self.inputs.assign(inputs);
        self.activations.assign(&activations);

        self.activation_fn.call(&activations)
    }

    fn back_prop(
        &mut self,
        actual: &Array1<f64>,
        target: &Array1<f64>,
        attached_layer: Option<&Layer>,
        cost: Box<dyn Cost>,
    ) {
        let prev_delta: Array1<f64>;

        match attached_layer {
            Some(layer) => prev_delta = layer.weights.t().dot(&layer.delta),
            _ => prev_delta = cost.prime(&actual, &target),
        };

        self.delta = self.activation_fn.prime(&self.activations) * &prev_delta;
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

impl Clone for Layer {
    fn clone(&self) -> Layer {
        Layer {
            weights: self.weights.to_owned(),
            biases: self.biases.to_owned(),
            inputs: self.inputs.to_owned(),
            activations: self.activations.to_owned(),
            delta: self.delta.to_owned(),
            neurons: self.neurons,
            activation_fn: self.activation_fn.box_clone(),
        }
    }
}

pub struct Network {
    // TODO do we only need input_layer and output_layer?
    layers: Vec<Layer>,
    cost: Box<dyn Cost>,
}

impl Network {
    pub fn new(cost: Box<dyn Cost>) -> Network {
        Network {
            layers: vec![],
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
        let prev_neurons = self.layers.last_mut().unwrap().neurons;

        self.layers
            .push(Layer::new(neurons, prev_neurons, activation_fn));
    }

    pub fn add_layer(
        &mut self,
        neurons: usize,
        inputs: Option<usize>,
        activation_fn: Box<dyn ActivationFn>,
    ) {
        match inputs {
            Some(inputs) => self.add_input_layer(neurons, inputs, activation_fn),
            _ => self.add_hidden_layer(neurons, activation_fn),
        }
    }

    pub fn fit(
        &mut self,
        inputs: &Vec<Array1<f64>>,
        outputs: &Vec<Array1<f64>>,
        optimizer: Box<dyn Optimizer>,
        epochs: usize,
    ) {
        for _ in 0..epochs {
            // TODO early stop condition
            let mut samples: Vec<usize> = (0..inputs.len()).collect();
            samples.shuffle(&mut thread_rng());

            for sample in samples {
                let network_output = self.predict(&inputs[sample]);
                let layers = self.layers.to_owned();
                for (i, layer) in self.layers.iter_mut().rev().enumerate() {
                    println!("i: {}", i);
                    let attached_layer = if i > 0 { Some(&layers[i - 1]) } else { None };
                    layer.back_prop(
                        &network_output,
                        &outputs[sample],
                        attached_layer,
                        self.cost.box_clone(),
                    );
                }

                for (i, layer) in self.layers.iter_mut().enumerate() {
                    layer.update(i, optimizer.box_clone());
                }
            }
        }
        // TODO return errors
    }

    pub fn predict(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let mut output: Array1<f64> = inputs.to_owned(); // clone();

        for layer in self.layers.iter_mut() {
            output = layer.feed_forward(&output);
        }
        output
    }
}
