use super::activations::ActivationFn;
use super::costs::Cost;

use ndarray::{Array, Array2};

use ndarray_rand::RandomExt;
use rand::{
    distributions::{Distribution, Uniform},
    prelude::ThreadRng
};

use serde::ser::{Serialize, SerializeStruct, Serializer};

/// Representation of a single Layer in the Network
pub struct Layer {
    /// Delta values computed using the first derivative of
    /// the Layer's activation function during backprop. Used
    /// to compute the gradient during the update stage
    pub delta: Array2<f64>,

    /// Input vector recorded during the feed-forward process
    pub inputs: Array2<f64>,

    /// Number of neurons, determines how many
    /// weights/biases are present
    pub neurons: usize,

    /// Matrix of weights (shape: neurons x inputs)
    weights: Array2<f64>,

    /// Vector of bias offsets
    biases: Array2<f64>,

    /// Activation values: (weights dot inputs) + biases
    activations: Array2<f64>,

    /// Function that determines the activation of individual neurons
    activation_fn: Box<dyn ActivationFn>,

    /// Dropout regularization chance
    dropout: Option<f32>,

    /// Row indices of neurons that have been dropped out
    /// temporarily during training
    dropped_neurons: Vec<usize>
}

impl Layer {
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases are present
    /// * `inputs` - Size of expected input vector
    /// * `activation_fn` - Function that determines the activation of individual neurons
    pub fn new(
        neurons: usize,
        inputs: usize,
        activation_fn: Box<dyn ActivationFn>,
        dropout: Option<f32>
    ) -> Layer {
        let weights: Array2<f64> = Array2::random((neurons, inputs), Uniform::new(0., 1.));
        let weights: Array2<f64> = weights / f64::sqrt(inputs as f64);

        let biases: Array2<f64> = Array2::random((neurons, 1), Uniform::new(0., 1.));

        let inputs: Array2<f64> = Array2::zeros((inputs, 1));
        let activations: Array2<f64> = Array2::zeros((neurons, 1));

        let delta: Array2<f64> = Array2::zeros((neurons, 1));
        let dropped_neurons: Vec<usize> = vec![];

        Layer {
            delta,
            inputs,
            neurons,
            weights,
            biases,
            activations,
            activation_fn,
            dropout,
            dropped_neurons
        }
    }

    /// Feedforward step for an individual Layer. Used for predicting outputs from a given input
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input vector to calculate activation values
    pub fn feed_forward(&mut self, inputs: &Array2<f64>, rng: &Option<ThreadRng>) -> Array2<f64> {
        let activations: Array2<f64> = self.weights.dot(inputs) + &self.biases;

        self.inputs.assign(inputs);
        self.activations.assign(&activations);

        let output: Array2<f64> = self.activation_fn.call(&activations);

        match self.dropout {
            Some(dropout) => {
                self.dropped_neurons.clear();
                self.map_output_with_dropout(output, dropout, rng)
            }
            None => output
        }
    }

    fn map_output_with_dropout(
        &mut self,
        output: Array2<f64>,
        dropout: f32,
        rng: &Option<ThreadRng>
    ) -> Array2<f64> {
        match rng {
            Some(mut some_rng) => {
                let mut index: usize = 0;
                let range = Uniform::new(0.0, 1.0);

                output.mapv(|el| {
                    let sample: f32 = range.sample(&mut some_rng);

                    let value = if sample < dropout {
                        self.dropped_neurons.push(index);
                        0.0
                    } else {
                        el
                    };

                    index += 1;
                    value
                })
            }
            None => output
        }
    }

    /// Backpropogation step where the deltas for each layer are calculated
    /// (do this step before gradient descent)
    ///
    /// # Arguments
    ///
    /// * `actual` - The predicted output produced by the network
    /// * `target` - The expected output value
    /// * `attached_layer` - The next layer in the network. Should be
    /// 'None' if 'self' is the layer that produces the final output
    /// * `cost` - The cost or loss function associated with the
    /// training setup
    pub fn back_prop(
        &mut self,
        actual: &Array2<f64>,
        target: &Array2<f64>,
        attached_layer: Option<Layer>,
        cost: Box<dyn Cost>
    ) {
        let attached_delta: Array2<f64>;

        match attached_layer {
            Some(layer) => attached_delta = layer.weights.t().dot(&layer.delta),
            _ => attached_delta = cost.prime(&actual, &target)
        };

        self.delta = self.activation_fn.prime(&self.activations) * &attached_delta;

        match self.dropout {
            Some(_) => {
                let zeros = Array::zeros(1);

                for dropped_neuron in self.dropped_neurons.iter() {
                    self.delta.row_mut(*dropped_neuron).assign(&zeros);
                }
            }
            None => ()
        }
    }

    /// Adjusts the weights and biases based on deltas calculated during gradient descent
    ///
    /// # Arguments
    ///
    /// * `delta_weights` - Change in the weight values
    /// * `delta_biases` - Change in the bias values
    pub fn update(&mut self, delta_weights: &Array2<f64>, delta_biases: &Array2<f64>) {
        let weights: Array2<f64> = &self.weights - delta_weights;
        let biases: Array2<f64> = &self.biases - delta_biases;

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
            activation_fn: self.activation_fn.clone(),
            dropout: self.dropout.clone(),
            dropped_neurons: self.dropped_neurons.clone()
        }
    }
}

impl Serialize for Layer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer
    {
        let mut s = serializer.serialize_struct("Layer", 2)?;
        s.serialize_field("weights", &self.weights)?;
        s.serialize_field("biases", &self.biases)?;

        s.end()
    }
}
