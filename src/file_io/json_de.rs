use crate::nn::functions::activation::{ActivationFn, LeakyReLU, ReLU, Sigmoid};
use crate::nn::functions::cost::{Cost, MSE};
use crate::nn::functions::encoder::{Encoder, OneHot};
use crate::nn::functions::metric::{Accuracy, Metric};
use crate::nn::functions::optimizer;
use crate::nn::functions::optimizer::{Adam, Optimizer, SGD};
use crate::nn::perceptron::Perceptron;
use ndarray::Array2;
use serde::Deserialize;
use serde_json::{Map, Value};

/// Deserialized values representing both input and output data in JSON
#[derive(Deserialize, Debug)]
struct DataDe {
    /// Training set input data
    train_inputs: Array2<f64>,

    /// Training set output data
    train_outputs: Array2<f64>,

    /// Validation set input data
    test_inputs: Array2<f64>,

    /// Validation set output data
    test_outputs: Array2<f64>,
}

/// Deserialized values representing a single Layer in JSON
#[derive(Deserialize, Debug, Clone)]
struct LayerDe {
    /// Number of neurons
    neurons: usize,

    /// Dropout chance (for regularization)
    dropout_rate: Option<f32>,

    /// Name of activation function
    activation: String,
}

/// Deserialized values representing the Optimizer in JSON
#[derive(Deserialize, Debug, Clone)]
struct OptimizerDe {
    /// Name of the optimization method
    name: String,

    /// Learning rate constant
    learning_rate: f64,

    /// Optional primary momentum constant
    beta1: Option<f64>,

    /// Optional secondary momentum constant
    beta2: Option<f64>,
}

///
#[derive(Deserialize, Debug, Clone)]
struct EncoderDe {
    /// Name of the Decoder
    name: String,

    /// Constructor arguments
    args: Map<String, Value>,
}

/// Deserialized values representing the evaluation Metric in JSON
#[derive(Deserialize, Debug, Clone)]
struct MetricDe {
    /// Name of the Metric
    name: String,

    /// Constructor arguments
    args: Map<String, Value>,
}

/// Deserialized values representing the Network setup in JSON
#[derive(Deserialize, Debug, Clone)]
struct NetworkDe {
    /// Cost function name
    cost: String,

    /// Hidden layer values
    layers: Vec<LayerDe>,

    /// Optimizer values
    optimizer: OptimizerDe,

    /// Output encoder
    encoder: EncoderDe,

    /// Metric values
    metric: MetricDe,

    /// Number of trianing epochs
    epochs: u64,
}

#[derive(Clone)]
/// Container for all deserialized data needed to train a network
pub struct NetworkDataDe {
    /// Training set input data
    pub train_inputs: Array2<f64>,

    /// Training set output data
    pub train_outputs: Array2<f64>,

    /// Validation set input data
    pub test_inputs: Array2<f64>,

    /// Validation set output data
    pub test_outputs: Array2<f64>,

    /// Network cost function
    pub cost: Box<dyn Cost>,

    /// Network evaluation method
    pub metric: Box<dyn Metric>,

    /// Gradient descent method
    pub optimizer: Box<dyn Optimizer>,

    /// Output encoder
    pub encoder: Box<dyn Encoder>,

    /// Number of trianing epochs
    pub epochs: u64,

    /// Deserailized paramaters for network creation
    network_de: NetworkDe
}

impl NetworkDataDe {
    /// # Arguments
    /// 
    /// * `data_json` - Raw contents of JSON file containing 
    /// training and validation data
    /// * `network_json` - Raw contents of JSON file containg
    /// network parameters
    pub fn from_json<'a>(
        data_json: &'a str,
        network_json: &'a str,
    ) -> Result<NetworkDataDe, &'static str> {
        // Deserialize raw file contents into struct values
        let data_de: DataDe = serde_json::from_str(data_json).unwrap();
        let network_de: NetworkDe = serde_json::from_str(network_json).unwrap();

        let cost: Box<dyn Cost> = match cost_from_str(network_de.cost.to_lowercase()) {
            Some(value) => value,
            None => return Err("Invalid cost function name"),
        };
        let metric: Box<dyn Metric> = match metric_from_str(&network_de.metric) {
            Some(value) => value,
            None => return Err("Invalid metric name"),
        };
        let encoder: Box<dyn Encoder> = match encoder_from_str(&network_de.encoder) {
            Some(value) => value,
            None => return Err("Invalid decoder name"),
        };
        let optimizer = match optimizer_from_str(&network_de.optimizer) {
            Some(value) => value,
            None => return Err("Invalid activation function name"),
        };

        Ok(NetworkDataDe {
            train_inputs: data_de.train_inputs,
            train_outputs: data_de.train_outputs,
            test_inputs: data_de.test_inputs,
            test_outputs: data_de.test_outputs,
            cost,
            metric,
            encoder,
            optimizer,
            epochs: network_de.epochs,
            network_de
        })
    }

    /// 
    pub fn create_network(&self) -> Result<Perceptron, &'static str> {
        let mut network = Perceptron::new();
        let input_shape: (usize, usize) = (self.train_inputs.ncols(), self.train_inputs.nrows());
        let mut input_shape: Option<(usize, usize)> = Some(input_shape);

        for layer in self.network_de.layers.iter() {
            let activation_fn: Box<dyn ActivationFn> =
                match activation_from_str(layer.activation.to_lowercase()) {
                    Some(value) => value,
                    None => return Err("Invalid activation function name"),
                };

            network.add_layer(layer.neurons, input_shape, activation_fn, layer.dropout_rate);
            if input_shape.is_some() {
                input_shape = None
            }
        }
        Ok(network)
    }
}

/// Create new 'Cost' object if the provided name 
/// matches an existing cost function
fn cost_from_str(name: String) -> Option<Box<dyn Cost>> {
    match name.as_str() {
        "mean squared error" | "mean_squared_error" | "mse" => Some(Box::new(MSE)),
        _ => None,
    }
}

/// Create new 'ActivationFn' object if the provided name 
/// matches an existing activation function
fn activation_from_str(name: String) -> Option<Box<dyn ActivationFn>> {
    match name.as_str() {
        "sigmoid" => Some(Box::new(Sigmoid)),
        "relu" => Some(Box::new(ReLU)),
        "leaky relu" | "leaky_relu" | "leakyrelu" => Some(Box::new(LeakyReLU)),
        _ => None,
    }
}

/// Create new 'Metric' object if the provided name 
/// matches an existing metric
fn metric_from_str(metric_de: &MetricDe) -> Option<Box<dyn Metric>> {
    match metric_de.name.to_lowercase().as_str() {
        "accuracy" | "acc" => Some(Box::new(Accuracy::new(&metric_de.args))),
        _ => None,
    }
}

/// Create new 'Encoder' object if the provided name 
/// matches an existing encoder
fn encoder_from_str(decode_de: &EncoderDe) -> Option<Box<dyn Encoder>> {
    match decode_de.name.to_lowercase().as_str() {
        "one hot" | "one_hot" | "onehot" => Some(Box::new(OneHot::new(&decode_de.args))),
        _ => None,
    }
}

/// Create new 'Optimizer' object if the provided name 
/// matches an existing optimization function
fn optimizer_from_str(optimizer_de: &OptimizerDe) -> Option<Box<dyn Optimizer>> {
    let beta1: f64 = match optimizer_de.beta1 {
        Some(beta1) => beta1,
        None => optimizer::DEFAULT_GAMMA,
    };

    let beta2: f64 = match optimizer_de.beta2 {
        Some(beta2) => beta2,
        None => optimizer::DEFAULT_BETA,
    };

    match optimizer_de.name.to_lowercase().as_str() {
        "stochastic gradient descent" | "gradient descent" | "sgd" => Some(Box::new(SGD::new(optimizer_de.learning_rate, beta1))),
        "adaptive momentum" | "adam" => Some(Box::new(Adam::new(
            optimizer_de.learning_rate,
            beta1,
            beta2,
        ))),
        _ => None,
    }
}
