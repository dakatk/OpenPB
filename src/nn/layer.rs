use super::activation::ActivationFn;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use serde::ser::{Serialize, SerializeStruct, Serializer};

/// Representation of a single Layer in the Network
#[derive(Clone)]
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
        input_shape: (usize, usize),
        activation_fn: Box<dyn ActivationFn>,
        dropout: Option<f32>
    ) -> Layer {
        let weights: Array2<f64> = Array2::random((neurons, input_shape.0), Uniform::new(0., 1.));
        let weights: Array2<f64> = weights / f64::sqrt(input_shape.0 as f64);

        let biases: Array2<f64> = Array2::random((neurons, 1), Uniform::new(0., 1.));

        let activations: Array2<f64> = Array2::zeros((neurons, input_shape.1));
        let delta: Array2<f64> = Array2::zeros((neurons, input_shape.1));
        let inputs: Array2<f64> = Array2::zeros(input_shape);

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
    pub fn feed_forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let activations: Array2<f64> = self.weights.dot(inputs) + &self.biases;
        let output: Array2<f64> = self.activation_fn.call(&activations);

        self.inputs.assign(inputs);
        self.activations.assign(&activations);

        match self.dropout {
            Some(dropout) => {
                self.dropped_neurons.clear();
                self.map_output_to_dropout(output, dropout)
            }
            None => output
        }
    }

    /// 
    pub fn predict(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let activations: Array2<f64> = self.weights.dot(inputs) + &self.biases;
        self.activation_fn.call(&activations)
    }

    /// 
    fn map_output_to_dropout(&mut self, mut output: Array2<f64>, dropout: f32) -> Array2<f64> {
        let range: Uniform<f32> = Uniform::new(0.0, 1.0);
        let zeros: Array1<f64> = Array1::zeros(output.ncols());

        let mut rng = thread_rng();

        for (i, mut row) in output.axis_iter_mut(Axis(0)).enumerate() {
            let sample: f32 = range.sample(&mut rng);
            if sample < dropout {
                self.dropped_neurons.push(i);
                row.assign(&zeros);
            }
        }
        output
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
    pub fn back_prop(&mut self, attached_layer: &Layer) {
        let attached_delta: Array2<f64> = attached_layer.weights.t().dot(&attached_layer.delta);
        self.back_prop_with_delta(&attached_delta);
    }

    /// 
    pub fn back_prop_with_delta(&mut self, delta: &Array2<f64>) {
        self.delta = self.activation_fn.prime(&self.activations) * delta;
        self.drop_neurons();
    }

    /// 
    fn drop_neurons(&mut self) {
        match self.dropout {
            Some(_) => {
                let zeros: Array1<f64> = Array1::zeros(self.delta.dim().1);
                for dropped_neuron in self.dropped_neurons.iter() {
                    self.delta.row_mut(*dropped_neuron).assign(&zeros);
                }
            }
            None => {}
        }
    }

    /// Adjusts the weights and biases based on deltas calculated during gradient descent
    ///
    /// # Arguments
    ///
    /// * `delta_weights` - Change in the weight values
    /// * `delta_biases` - Change in the bias values
    pub fn update(&mut self, delta_weights: &Array2<f64>, delta_biases: &Array2<f64>, input_rows: usize) {
        let delta_weights: Array2<f64> = delta_weights / (input_rows as f64);
        let delta_biases: f64 = delta_biases.sum() / (input_rows as f64);

        let weights: Array2<f64> = &self.weights - delta_weights;
        let biases: Array2<f64> = &self.biases - delta_biases;

        self.weights.assign(&weights);
        self.biases.assign(&biases);
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
