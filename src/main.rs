// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod nn;
mod parse_json;

use nn::network::Network;

use serde_json::Value;

use std::env;
use std::fs;

#[doc(hidden)]
fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();

    let filename = match args.len() {
        2 => args[1].to_string(),
        _ => return Err("Incorrect number of args".to_string())
    };

    let file_contents: String;

    match fs::read_to_string(&filename) {
        Ok(result) => file_contents = result,
        Err(_) => return Err(format!("'{}' missing or corrupted", filename).to_string())
    };

    let network_json: Value = serde_json::from_str(&file_contents).unwrap();

    if !parse_json::has_keys(
        &network_json,
        vec![
            "input",
            "output",
            "cost",
            "hidden_layers",
            "optimizer",
            "metric",
            "epochs",
        ]
    ) {
        return Err("Invalid JSON file (missing required keys)".to_string());
    }

    let input_json = network_json["input"].as_object().unwrap();
    let output_json = network_json["output"].as_object().unwrap();
    let cost_json = network_json["cost"].as_str().unwrap();
    let layers_json = network_json["hidden_layers"].as_array().unwrap();
    let optimizer_json = network_json["optimizer"].as_object().unwrap();
    let metric_json = network_json["metric"].as_object().unwrap();
    let epochs_json = network_json["epochs"].as_u64().unwrap();

    let mut network = match parse_json::get_cost_fn(cost_json.to_string()) {
        Ok(cost) => Network::new(cost),
        Err(msg) => return Err(msg)
    };

    let input = match parse_json::get_input_data(input_json) {
        Ok(input) => input,
        Err(msg) => return Err(msg)
    };

    let output = match parse_json::get_output_data(output_json) {
        Ok(output) => output,
        Err(msg) => return Err(msg)
    };

    network.add_layer(
        input.neurons as usize,
        Some(input.size as usize),
        input.activation
    );

    for layer in layers_json.iter() {
        let layer_values = match parse_json::get_layer(layer) {
            Ok(layer) => layer,
            Err(msg) => return Err(msg)
        };

        network.add_layer(layer_values.neurons as usize, None, layer_values.activation);
    }

    network.add_layer(output.size as usize, None, output.activation);

    let optimizer = match parse_json::get_optimizer(optimizer_json) {
        Ok(optimizer) => optimizer,
        Err(msg) => return Err(msg)
    };

    let metric = match parse_json::get_metric(metric_json) {
        Ok(metric) => metric,
        Err(msg) => return Err(msg)
    };

    network.fit(&input.data, &output.data, optimizer, metric, epochs_json);

    for (input, output) in input.data.iter().zip(output.data) {
        println!("{}: {} {}", input, network.predict(input), output);
    }

    Ok(())
}
