use crate::nn::{activations::{ActivationFn, LeakyReLU, ReLU, Sigmoid}, optimizers};
use crate::nn::costs::{Cost, MSE};
use crate::nn::metrics::{Accuracy, Metric};
use crate::nn::network::Network;
use crate::nn::optimizers::{Adam, Optimizer, SGD};

use ndarray::Array2;

use serde::Deserialize;
use serde_json::{Map, Value};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

/// Deserialized values representing the input data in JSON
#[derive(Deserialize, Debug)]
struct InputDe {
    /// Number of neurons
    neurons: usize,

    /// Name of activation function
    activation: String,

    /// Size of each input vector
    size: usize,

    /// Dropout chance (for regularization)
    dropout: Option<f32>,

    /// All input vectors
    data: Vec<Array2<f64>>
}

/// Deserialized values representing the output data in JSON
#[derive(Deserialize, Debug)]
struct OutputDe {
    /// Name of activation function
    activation: String,

    /// Size of each output vector
    size: usize,

    /// All output vectors
    data: Vec<Array2<f64>>
}

/// Deserialized values representing both input and output data in JSON
#[derive(Deserialize, Debug)]
struct DataDe {
    /// Input data
    inputs: InputDe,

    /// Output data
    outputs: OutputDe
}

/// Deserialized values representing a single Layer in JSON
#[derive(Deserialize, Debug)]
struct LayerDe {
    /// Number of neurons
    neurons: usize,

    /// Dropout chance (for regularization)
    dropout: Option<f32>,

    /// Name of activation function
    activation: String
}

/// Deserialized values representing the Optimizer in JSON
#[derive(Deserialize, Debug)]
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

/// Deserialized values representing the evaluation Metric in JSON
#[derive(Deserialize, Debug)]
struct MetricDe {
    /// Name of the Metric
    name: String,

    /// Any arguments, if needed
    args: Map<String, Value>
}

/// Deserialized values representing the Network setup in JSON
#[derive(Deserialize, Debug)]
struct NetworkDe {
    /// Cost function name
    cost: String,

    /// Hidden layer values
    hidden_layers: Vec<LayerDe>,

    /// Optimizer values
    optimizer: OptimizerDe,

    /// Metric values
    metric: MetricDe,

    /// Number of trianing epochs
    epochs: u64
}

pub struct NetworkDataDe {
    pub network: Network,
    pub inputs: Vec<Array2<f64>>,
    pub outputs: Vec<Array2<f64>>,
    pub metric: Box<dyn Metric>,
    pub optimizer: Box<dyn Optimizer>,
    pub epochs: u64
}

impl NetworkDataDe {
    pub fn from_json<'a>(
        data_json: &'a str,
        network_json: &'a str
    ) -> Result<NetworkDataDe, &'static str> {
        // TODO could this be optimized more?
        let data_values: DataDe = serde_json::from_str(data_json).unwrap();
        let network_values: NetworkDe = serde_json::from_str(network_json).unwrap();

        let input_values: InputDe = data_values.inputs;
        let output_values: OutputDe = data_values.outputs;

        let cost: Box<dyn Cost> = match cost_from_str(network_values.cost.to_lowercase()) {
            Some(value) => value,
            None => return Err("Invalid cost function name")
        };

        let mut network = Network::new(cost);

        let input_activation = match activation_from_str(input_values.activation.to_lowercase()) {
            Some(value) => value,
            None => return Err("Invalid activation function name")
        };

        network.add_layer(
            input_values.neurons,
            Some(input_values.size),
            input_activation,
            input_values.dropout
        );

        for layer in network_values.hidden_layers.iter() {
            let layer_activation = match activation_from_str(layer.activation.to_lowercase()) {
                Some(value) => value,
                None => return Err("Invalid activation function name")
            };

            network.add_layer(layer.neurons, None, layer_activation, layer.dropout);
        }

        let output_activation = match activation_from_str(output_values.activation.to_lowercase()) {
            Some(value) => value,
            None => return Err("Invalid activation function name")
        };

        network.add_layer(output_values.size, None, output_activation, None);

        let metric = match metric_from_str(network_values.metric) {
            Some(value) => value,
            None => return Err("Invalid metric name")
        };

        let optimizer = match optimizer_from_str(network_values.optimizer) {
            Some(value) => value,
            None => return Err("Invalid activation function name")
        };

        Ok(NetworkDataDe {
            network,
            inputs: input_values.data,
            outputs: output_values.data,
            metric,
            optimizer,
            epochs: network_values.epochs
        })
    }
}

fn cost_from_str(name: String) -> Option<Box<dyn Cost>> {
    match name.as_str() {
        "mse" => Some(Box::new(MSE)),
        _ => None
    }
}

fn activation_from_str(name: String) -> Option<Box<dyn ActivationFn>> {
    match name.as_str() {
        "sigmoid" => Some(Box::new(Sigmoid)),
        "relu" => Some(Box::new(ReLU)),
        "leakyrelu" => Some(Box::new(LeakyReLU)),
        _ => None
    }
}

fn metric_from_str(metric_data: MetricDe) -> Option<Box<dyn Metric>> {
    match metric_data.name.to_lowercase().as_str() {
        "accuracy" => Some(Box::new(Accuracy::new(&metric_data.args))),
        _ => None
    }
}

fn optimizer_from_str(optimizer_data: OptimizerDe) -> Option<Box<dyn Optimizer>> {
    let beta1 = match optimizer_data.beta1 {
        Some(beta1) => beta1,
        None => optimizers::DEFAULT_BETA_1
    };

    let beta2 = match optimizer_data.beta1 {
        Some(beta2) => beta2,
        None => optimizers::DEFAULT_BETA_2
    };

    match optimizer_data.name.to_lowercase().as_str() {
        "sgd" => Some(Box::new(SGD::new(optimizer_data.learning_rate, beta1))),
        "adam" => Some(Box::new(Adam::new(optimizer_data.learning_rate, beta1, beta2))),
        _ => None
    }
}

/// Save internal values (weights and biases) from each layer of a network
///
/// # Arguments
///
/// * `network` - Network object to be serialized
/// * `filename` - JSON file to write serialized values to
pub fn save_layer_values(network: &Network, filename: &str) -> Result<(), String> {
    println!("\nAttempting to write to {}...", filename);

    let mut file = match File::create(&Path::new(filename)) {
        Ok(file) => file,
        Err(msg) => return Err(format!("Failed to create file {}: {}", filename, msg))
    };

    let network_ser = serde_json::to_string_pretty(&network).unwrap();

    match file.write_all(network_ser.as_bytes()) {
        Ok(_) => {
            println!("Success!");
            Ok(())
        }
        Err(msg) => Err(msg.to_string())
    }
}
