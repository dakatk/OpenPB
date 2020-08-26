use crate::nn::activations::*;
use crate::nn::costs::*;
use crate::nn::metrics::*;
use crate::nn::optimizers::*;

use ndarray::Array1;

use serde_json::{Map, Value};

/// 
pub struct LayerValues {

    /// 
    pub neurons: u64,

    /// 
    pub activation: Box<dyn ActivationFn>
}

/// 
pub struct InputValues {

    /// 
    pub neurons: u64,

    /// 
    pub activation: Box<dyn ActivationFn>,

    /// 
    pub size: u64,

    /// 
    pub data: Vec<Array1<f64>>
}

/// 
pub struct OutputValues {

    /// 
    pub activation: Box<dyn ActivationFn>,

    /// 
    pub size: u64,

    /// 
    pub data: Vec<Array1<f64>>
}

/// 
pub fn has_keys(json: &Value, keys: Vec<&str>) -> bool {
    for key in keys {
        match json.get(key) {
            Some(_) => continue,
            None => return false
        }
    }

    true
}

/// 
pub fn get_input_data(args: &Map<String, Value>) -> Result<InputValues, String> {
    let neurons = match args["neurons"].as_u64() {
        Some(neurons) => neurons,
        None => return Err("Missing field 'neurons' from input".to_string())
    };

    let activation = match args["activation"].as_str() {
        Some(activation) => activation,
        None => return Err("Missing field 'activation' from layer".to_string())
    };

    let activation_fn: Box<dyn ActivationFn> = match activation.to_lowercase().as_str() {
        "sigmoid" => Box::new(Sigmoid),
        "relu" => Box::new(ReLU),
        _ => return Err("Invalid activation function name".to_string())
    };

    let size = match args["size"].as_u64() {
        Some(size) => size,
        None => return Err("Missing field 'size' from input".to_string())
    };

    let data = match args["data"].as_array() {
        Some(data) => data,
        None => return Err("Missing field 'data' from input".to_string())
    };

    Ok(InputValues {
        neurons: neurons,
        activation: activation_fn,
        size: size,
        data: values_to_f64_array(data)
    })
}

/// 
pub fn get_output_data(args: &Map<String, Value>) -> Result<OutputValues, String> {
    let activation = match args["activation"].as_str() {
        Some(activation) => activation,
        None => return Err("Missing field 'activation' from layer".to_string())
    };

    let activation_fn: Box<dyn ActivationFn> = match activation.to_lowercase().as_str() {
        "sigmoid" => Box::new(Sigmoid),
        "relu" => Box::new(ReLU),
        _ => return Err("Invalid activation function name".to_string())
    };

    let size = match args["size"].as_u64() {
        Some(size) => size,
        None => return Err("Missing field 'size' from input".to_string())
    };

    let data = match args["data"].as_array() {
        Some(data) => data,
        None => return Err("Missing field 'data' from input".to_string())
    };

    Ok(OutputValues {
        activation: activation_fn,
        size: size,
        data: values_to_f64_array(data)
    })
}

#[doc(hidden)]
fn values_to_f64_array(values: &Vec<Value>) -> Vec<Array1<f64>> {
    fn value_vec_to_f64_vec(value: &Vec<Value>) -> Vec<f64> {
        value.into_iter().map(|el| el.as_f64().unwrap()).collect()
    }

    values
        .into_iter()
        .map(|el| {
            let as_vec = value_vec_to_f64_vec(el.as_array().unwrap());

            Array1::from(as_vec)
        })
        .collect()
}

/// 
pub fn get_layer(args: &Value) -> Result<LayerValues, String> {
    let neurons = match args["neurons"].as_u64() {
        Some(neurons) => neurons,
        None => return Err("Missing field 'neurons' from layer".to_string())
    };

    let activation = match args["activation"].as_str() {
        Some(activation) => activation,
        None => return Err("Missing field 'activation' from layer".to_string())
    };

    let activation_fn: Box<dyn ActivationFn> = match activation.to_lowercase().as_str() {
        "sigmoid" => Box::new(Sigmoid),
        "relu" => Box::new(ReLU),
        _ => return Err("Invalid activation function name".to_string())
    };

    Ok(LayerValues {
        neurons: neurons,
        activation: activation_fn
    })
}

/// 
pub fn get_cost_fn(arg: String) -> Result<Box<dyn Cost>, String> {
    match arg.to_lowercase().as_str() {
        "mse" => Ok(Box::new(MSE)),
        _ => Err("Invalid cost function name".to_string())
    }
}

/// 
pub fn get_optimizer(args: &Map<String, Value>) -> Result<Box<dyn Optimizer>, String> {
    let name = match args["name"].as_str() {
        Some(name) => name,
        None => return Err("Missing field 'name' from optimizer".to_string())
    };

    let learning_rate = match args["learning_rate"].as_f64() {
        Some(learning_rate) => learning_rate,
        None => return Err("Missing field 'learning_rate' from optimizer".to_string())
    };

    match name.to_lowercase().as_str() {
        "sgd" => Ok(Box::new(SGD::new(learning_rate))),
        "adam" => Ok(Box::new(Adam::new(learning_rate))),
        _ => Err("Invalid optimizer name".to_string())
    }
}

/// 
pub fn get_metric(args: &Map<String, Value>) -> Result<Box<dyn Metric>, String> {
    let name = match args["name"].as_str() {
        Some(name) => name,
        None => return Err("Missing field 'name' from metric".to_string())
    };

    let params = match args["args"].as_object() {
        Some(params) => params,
        None => return Err("Missing field 'args' from metric".to_string())
    };

    match name.to_lowercase().as_str() {
        "accuracy" => Ok(Box::new(Accuracy::new(params))),
        _ => Err("Invalid metric name".to_string())
    }
}
