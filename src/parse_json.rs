use crate::nn::activations::{ActivationFn, LeakyReLU, ReLU, Sigmoid};
use crate::nn::costs::{Cost, MSE};
use crate::nn::metrics::{Accuracy, Metric};
use crate::nn::network::Network;
use crate::nn::optimizers::{Adam, Optimizer, SGD};

use ndarray::Array1;

use serde::Deserialize;
use serde_json::{Map, Value};

/// Deserialized values representing the input data in JSON
#[derive(Deserialize)]
struct InputDe {
    /// Number of neurons
    neurons: usize,

    /// Name of activation function
    activation: String,

    /// Size of each input vector
    size: usize,

    /// All input vectors
    data: Vec<Array1<f64>>
}

/// Deserialized values representing the output data in JSON
#[derive(Deserialize)]
struct OutputDe {
    /// Name of activation function
    activation: String,

    /// Size of each output vector
    size: usize,

    /// All output vectors
    data: Vec<Array1<f64>>
}

/// Deserialized values representing both input and output data in JSON
#[derive(Deserialize)]
struct DataDe {
    /// Input data
    inputs: InputDe,

    /// Output data
    outputs: OutputDe
}

/// Deserialized values representing a single Layer in JSON
#[derive(Deserialize)]
struct LayerDe {
    /// Number of neurons
    neurons: usize,

    /// Name of activation function
    activation: String
}

/// Deserialized values representing the Optimizer in JSON
#[derive(Deserialize)]
struct OptimizerDe {
    /// Name of the optimization method
    name: String,

    /// Learning rate constant
    learning_rate: f64
}

/// Deserialized values representing the evaluation Metric in JSON
#[derive(Deserialize)]
struct MetricDe {
    /// Name of the Metric
    name: String,

    /// Any arguments, if needed
    args: Map<String, Value>
}

/// Deserialized values representing the Network setup in JSON
#[derive(Deserialize)]
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
    pub inputs: Vec<Array1<f64>>,
    pub outputs: Vec<Array1<f64>>,
    pub metric: Box<dyn Metric>,
    pub optimizer: Box<dyn Optimizer>,
    pub epochs: u64
}

impl NetworkDataDe {
    pub fn from_json<'a>(
        data_json: &'a str,
        network_json: &'a str
    ) -> Result<NetworkDataDe, &'static str> {
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
            input_activation
        );

        for layer in network_values.hidden_layers.iter() {
            let layer_activation = match activation_from_str(layer.activation.to_lowercase()) {
                Some(value) => value,
                None => return Err("Invalid activation function name")
            };

            network.add_layer(layer.neurons, None, layer_activation);
        }

        let output_activation = match activation_from_str(output_values.activation.to_lowercase()) {
            Some(value) => value,
            None => return Err("Invalid activation function name")
        };

        network.add_layer(output_values.size, None, output_activation);

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
    match optimizer_data.name.to_lowercase().as_str() {
        "sgd" => Some(Box::new(SGD::new(optimizer_data.learning_rate))),
        "adam" => Some(Box::new(Adam::new(optimizer_data.learning_rate))),
        _ => None
    }
}
