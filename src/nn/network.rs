use crate::nn::activations::ActivationFn;
use crate::nn::costs::Cost;
use crate::nn::metrics::Metric;
use crate::nn::optimizers::Optimizer;

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use serde::ser::{Serialize, SerializeStruct, Serializer};

/// A single Layer in the Network
struct Layer {
    /// Matrix of weights (shape: neurons x inputs)
    weights: Array2<f64>,

    /// Vector of bias offsets
    biases: Array1<f64>,

    /// Input vector recorded during the feed-forward process
    inputs: Array1<f64>,

    /// Activation values: (weights dot inputs) + biases
    activations: Array1<f64>,

    /// Delta values computed using the first derivative of
    /// the Layer's activation function during backprop. Used
    /// to compute the gradient during the update stage
    delta: Array1<f64>,

    /// Number of neurons, determines how many
    /// weights/biases are present
    neurons: usize,

    /// Function that determines the activation of individual neurons
    activation_fn: Box<dyn ActivationFn>
}

impl Layer {
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases are present
    /// * `inputs` - Size of expected input vector
    /// * `activation_fn` - Function that determines the activation of individual neurons
    fn new(neurons: usize, inputs: usize, activation_fn: Box<dyn ActivationFn>) -> Layer {
        Layer {
            weights: Array::random((neurons, inputs), Uniform::new(0., 1.)),
            biases: Array::random(neurons, Uniform::new(0., 1.)),
            inputs: Array::zeros(inputs),
            activations: Array::zeros(neurons),
            delta: Array::zeros(neurons),
            neurons,
            activation_fn
        }
    }

    /// Feedforward step for an individual Layer. Used for predicting outputs from a given input
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input vector to calculate activation values
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
        attached_layer: Option<Layer>,
        cost: Box<dyn Cost>
    ) {
        let prev_delta: Array1<f64>;

        match attached_layer {
            Some(layer) => prev_delta = layer.weights.t().dot(&layer.delta),
            _ => prev_delta = cost.prime(&actual, &target)
        };

        self.delta = self.activation_fn.prime(&self.activations) * &prev_delta;
    }

    /// Adjusts the weights/biases of the Layer based on the calculated delta,
    /// input vector, and chosen Optimization method
    ///
    /// # Arguments
    ///
    /// * `index` - Numeric index of the current Layer's placement within the Network
    /// * `optimizer` - Optimization method used to perform perform gradient descent
    fn update<'a>(&mut self, index: usize, optimizer: &mut (dyn Optimizer + 'a)) {
        let delta: Array2<f64> = self.delta.clone().insert_axis(Axis(1));
        let inputs: Array2<f64> = self.inputs.clone().insert_axis(Axis(0));
        let gradient: Array2<f64> = delta.dot(&inputs);

        let delta_weights = -optimizer.delta(index, &gradient);
        let delta_biases = -optimizer.learning_rate() * &self.delta;

        let weights: Array2<f64> = delta_weights + &self.weights;
        let biases: Array1<f64> = delta_biases + &self.biases;

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
            activation_fn: self.activation_fn.clone()
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

pub struct Network {
    /// Input, hidden, and output layers. Each layer is considered
    /// to be 'connected' to the next one in the list
    layers: Vec<Layer>,

    /// Loss function for error reporting/backprop
    cost: Box<dyn Cost>
}

impl Network {
    /// # Arguments:
    ///
    /// * `cost` - Loss function for error reporting/backprop
    pub fn new(cost: Box<dyn Cost>) -> Network {
        Network {
            layers: vec![],
            cost
        }
    }

    /// Creates a new layer and adds it to the Network. Used only for the
    /// first layer added, which is treated as the input layer
    ///
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases
    /// are present in the new Layer
    /// * `inputs` - Size of expected the Layer's input vector
    /// * `activation_fn` - Function that determines the activation of individual neurons
    fn add_input_layer(
        &mut self,
        neurons: usize,
        inputs: usize,
        activation_fn: Box<dyn ActivationFn>
    ) {
        self.layers.push(Layer::new(neurons, inputs, activation_fn));
    }

    /// Same as `add_input_layer`, but used for any other layer after. The number of
    /// inputs for a hidden layer is equal to the number of neurons in the preceding layer
    ///
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases
    /// are present in the new Layer
    /// * `activation_fn` - Function that determines the activation of individual neurons
    fn add_hidden_layer(&mut self, neurons: usize, activation_fn: Box<dyn ActivationFn>) {
        let prev_neurons = self.layers.last_mut().unwrap().neurons;

        self.layers
            .push(Layer::new(neurons, prev_neurons, activation_fn));
    }

    /// Add a Layer to the next open spot in the Network's structure. This function
    /// also dynamically expands the Network's overall size
    ///
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases
    /// are present in the new Layer
    /// * `inputs` (optional) - Size of expected the Layer's input vector
    /// * `activation_fn` - Function that determines the activation of individual neurons
    pub fn add_layer(
        &mut self,
        neurons: usize,
        inputs: Option<usize>,
        activation_fn: Box<dyn ActivationFn>
    ) {
        match inputs {
            Some(inputs) => self.add_input_layer(neurons, inputs, activation_fn),
            _ => self.add_hidden_layer(neurons, activation_fn)
        }
    }

    /// Trains the entire Network for a specified number of cycles. Training is
    /// stopped when the given metric is satisfied based on the input/output
    /// sets provided
    ///
    /// # Arguments
    ///
    /// * `inputs` - Set of all input vectors to train the Network on
    /// * `outputs` - Set of corresponding output vectors
    /// * `optimizer` - Optimization method used to perform perform gradient descent
    /// * `metric` - Decides when the Network is performing 'good enough'
    /// on the provided data
    /// * `epochs` - Maximum number of training cycles
    pub fn fit(
        &mut self,
        inputs: &[Array1<f64>],
        outputs: &[Array1<f64>],
        mut optimizer: Box<dyn Optimizer>,
        metric: Box<dyn Metric>,
        epochs: u64
    ) -> Vec<Array1<f64>> {
        for epoch in 1..=epochs {
            let mut early_stop = true;
            let samples = optimizer.next(inputs.len());

            for sample in samples {
                let network_output: Array1<f64> = self.predict(&inputs[sample]);
                let len: usize = self.layers.to_owned().len();

                let mut attached_layer: Option<Layer>;

                for i in (0..len).rev() {
                    {
                        attached_layer = if i < len - 1 {
                            let layer = self.layers[i + 1].clone();
                            Some(layer)
                        } else {
                            None
                        };
                    }

                    self.layers[i].back_prop(
                        &network_output,
                        &outputs[sample],
                        attached_layer,
                        self.cost.clone()
                    );
                }

                for (i, layer) in self.layers.iter_mut().enumerate() {
                    layer.update(i, &mut *optimizer);
                }

                if !metric.call(&network_output, &outputs[sample]) {
                    early_stop = false;
                }
            }

            if early_stop {
                println!("Training ended on epoch {}", epoch);
                break;
            }
        }

        let mut errors: Vec<Array1<f64>> = vec![];

        for (input, output) in inputs.iter().zip(outputs) {
            let network_output = self.predict(input);
            let error = self.cost.prime(&network_output, output);

            errors.push(error);
        }

        errors
    }

    /// Performs the feedforward step for all Layers to return the
    /// Network's prediction for a given input vector
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input vector
    pub fn predict(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let mut output: Array1<f64> = inputs.to_owned();

        for layer in self.layers.iter_mut() {
            output = layer.feed_forward(&output);
        }

        output
    }
}
