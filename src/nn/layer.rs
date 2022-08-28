use super::functions::activation::ActivationFn;
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
    pub deltas: Option<Array2<f64>>,

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
    activations: Option<Array2<f64>>,

    /// Function that determines the activation of individual neurons
    activation_fn: Box<dyn ActivationFn>,

    /// Dropout regularization chance
    dropout: Option<f32>,

    /// Row indices of neurons that have been dropped out
    /// temporarily during training
    dropped_neurons: Vec<usize>,
}

impl Layer {
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases are present
    /// * `inputs` - Size of expected input vector
    /// * `activation_fn` - Function that determines the activation of individual neurons
    /// * `dropout` - Optional rate for randomly excluding neurons during each training cycle
    pub fn new(
        neurons: usize,
        input_shape: (usize, usize),
        activation_fn: Box<dyn ActivationFn>,
        dropout: Option<f32>,
    ) -> Layer {
        // Weights and biases are initialized randomly
        // in the range [-0.5, 0.5)
        let distribution: Uniform<f64> = Uniform::new(-0.5, 0.5);

        // Create weights matrix
        let weights: Array2<f64> = Array2::random((neurons, input_shape.0), distribution);
        // Scaling the weights by the sqrt of the number of nodes
        // helps to reduce the problem of disappearing gradient
        let weights: Array2<f64> = weights / f64::sqrt(input_shape.1 as f64);

        // Create biases matrix
        let biases: Array2<f64> = Array2::random((neurons, 1), distribution);

        // Stored inputs initialized to zero
        let inputs: Array2<f64> = Array2::zeros(input_shape);

        Layer {
            deltas: None,
            inputs,
            neurons,
            weights,
            biases,
            activations: None,
            activation_fn,
            dropout,
            dropped_neurons: vec![],
        }
    }

    /// Feedforward step for an individual Layer. Used for predicting outputs from a given input
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix of input vectors (outputs from previous layer)
    pub fn feed_forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let activations: Array2<f64> = self.weights.dot(inputs) + &self.biases;
        let outputs: Array2<f64> = self.activation_fn.call(&activations);

        self.inputs = inputs.clone();
        self.activations = Some(activations);

        match self.dropout {
            Some(dropout) => {
                self.dropped_neurons.clear();
                self.map_output_to_dropout(outputs, dropout)
            }
            None => outputs,
        }
    }

    /// Same as `feed_forward`, but dropout isn't applied and internal values aren't
    /// saved. This function is meant to get predictions from a fully-trained network
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix of input vectors (outputs from previous layer)
    pub fn predict(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let activations: Array2<f64> = self.weights.dot(inputs) + &self.biases;
        self.activation_fn.call(&activations)
    }

    /// Randomly choose dropped neurons for the current training cycle and
    /// change the respective output vectors to zeroed vectors of the same size
    ///
    /// # Arguments
    ///
    /// * `outputs` - Matrix of output vectors from last feedforward pass for
    /// the current layer
    /// * `dropout` - Rate at which neurons are dropped during training
    fn map_output_to_dropout(&mut self, mut outputs: Array2<f64>, dropout: f32) -> Array2<f64> {
        let range: Uniform<f32> = Uniform::new(0.0, 1.0);
        let zeros: Array1<f64> = Array1::zeros(outputs.ncols());

        let mut rng = thread_rng();

        for (i, mut row) in outputs.axis_iter_mut(Axis(0)).enumerate() {
            let sample: f32 = range.sample(&mut rng);
            if sample < dropout {
                self.dropped_neurons.push(i);
                row.assign(&zeros);
            }
        }
        outputs
    }

    /// Backpropogation step where the deltas for each layer are calculated
    /// (do this step before gradient descent)
    ///
    /// # Arguments
    ///
    /// * `actual` - The predicted output produced by the network
    /// * `target` - The expected output value
    /// * `attached_layer` - The next layer in the network
    /// * `cost` - The cost or loss function associated with the
    /// training setup
    pub fn back_prop(&mut self, attached_layer: &Layer) {
        let attached_deltas: &Array2<f64> = match &attached_layer.deltas {
            Some(attached_deltas) => attached_deltas,
            None => panic!("Deltas not calculated for attached layer"),
        };
        let next_deltas: Array2<f64> = attached_layer.weights.t().dot(attached_deltas);
        self.back_prop_with_deltas(&next_deltas);
    }

    /// Computes current layer's delta values from attached layer's deltas
    ///
    /// # Arguments
    ///
    /// * `attached_deltas` - Attached layer's deltas (assumed to have
    /// already been computed)
    pub fn back_prop_with_deltas(&mut self, attached_deltas: &Array2<f64>) {
        let activations: &Array2<f64> = match &self.activations {
            Some(activations) => activations,
            None => panic!("Error: back prop run before feed forward"),
        };
        let deltas: Array2<f64> = self.activation_fn.prime(activations) * attached_deltas;
        self.deltas = Some(deltas);
        self.drop_deltas();
    }

    /// Remove deltas relative to which neurons have been dropped
    /// during the latest training cycle
    fn drop_deltas(&mut self) {
        let deltas: &mut Array2<f64> = match &mut self.deltas {
            Some(deltas) => deltas,
            None => panic!("Can't drop deltas if deltas haven't been calculated"),
        };
        match self.dropout {
            Some(_) => {
                let zeros: Array1<f64> = Array1::zeros(deltas.ncols());
                for dropped_neuron in self.dropped_neurons.iter() {
                    deltas.row_mut(*dropped_neuron).assign(&zeros);
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
    pub fn update(
        &mut self,
        delta_weights: &Array2<f64>,
        delta_biases: &Array2<f64>,
        input_rows: usize,
    ) {
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
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("Layer", 2)?;

        // Only weights and biases are serialized
        s.serialize_field("weights", &self.weights)?;
        s.serialize_field("biases", &self.biases)?;
        s.end()
    }
}
