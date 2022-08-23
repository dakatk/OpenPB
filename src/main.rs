// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod file_io;
mod nn;

use clap::{App, Arg, ArgMatches};
use file_io::parse_json::NetworkDataDe;
use file_io::save_output;
use ndarray::Array2;
use nn::metric::Metric;
use nn::optimizer::Optimizer;
use nn::perceptron::Perceptron;
use std::fs;
use std::time::SystemTime;

#[doc(hidden)]
fn main() -> Result<(), String> {
    let args: ArgMatches = App::new("Open Neural Network Benchmarker (ONNB)")
        .version("0.1")
        .author("Dusten Knull <dakatk97@gmail.com>")
        .arg(
            Arg::with_name("data")
                .short('d')
                .long("data")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(true)
                .help("JSON file with input and output vectors"),
        )
        .arg(
            Arg::with_name("network")
                .short('n')
                .long("network")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(true)
                .help("JSON file with network structure and hyperparameters"),
        )
        .arg(
            Arg::with_name("output")
                .short('o')
                .long("output")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(false)
                .help("JSON file where weights and biases are stored (optional)"),
        )
        .get_matches();

    let network_filename: &str = args.value_of("network").unwrap();
    let data_filename: &str = args.value_of("data").unwrap();

    let network_json = match fs::read_to_string(&network_filename) {
        Ok(result) => result,
        _ => return Err(format!("File {network_filename} missing or corrupted")),
    };
    let data_json = match fs::read_to_string(&data_filename) {
        Ok(result) => result,
        _ => return Err(format!("File {data_filename} missing or corrupted")),
    };

    match NetworkDataDe::from_json(&data_json, &network_json) {
        Ok(mut result) => train_from_json(&mut result, &args),
        Err(error) => Err(error.to_string()),
    }
}

/// Train network with deserailzed JSON data
fn train_from_json(de_data: &mut NetworkDataDe, args: &ArgMatches) -> Result<(), String> {
    let now: SystemTime = SystemTime::now();
    let network: &mut Perceptron = &mut de_data.network;
    
    println!("Network initialized, tarting training cycle...\n");

    let optimizer: &mut dyn Optimizer = de_data.optimizer.as_mut();
    let metric: &dyn Metric = de_data.metric.as_ref();

    let training_set: (Array2<f64>, Array2<f64>) = (
        de_data.train_inputs.t().to_owned(),
        de_data.train_outputs.to_owned(),
    );
    let validation_set: (Array2<f64>, Array2<f64>) = (
        de_data.test_inputs.t().to_owned(),
        de_data.test_outputs.to_owned(),
    );

    // TODO Multi-threading (train and save data per thread)
    network.fit(
        &training_set,
        &validation_set,
        optimizer,
        metric,
        de_data.cost.as_ref(),
        de_data.encoder.as_ref(),
        de_data.epochs,
        false,
    );

    let elapsed: f32 = now.elapsed().unwrap().as_secs_f32();
    println!("Finished after {elapsed} seconds\n");

    let prediction: Array2<f64> = network.predict(&validation_set.0, de_data.encoder.as_ref());
    let expected: &Array2<f64> = &validation_set.1;

    println!(
        "{}: {} (passed: {})\n",
        metric.label(),
        metric.check(&prediction, expected),
        metric.value(&prediction, expected)
    );
    println!("Final prediction for validation set:\n{prediction}\n");
    println!("Expected outputs from validation set:\n{expected}\n");

    save_output::save_to_dir(&args, &network)
}
