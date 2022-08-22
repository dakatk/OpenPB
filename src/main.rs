// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod nn;
mod parse_json;

use clap::{App, Arg, ArgMatches};
use ndarray::Array2;
use nn::perceptron::Perceptron;
use nn::optimizer::Optimizer;
use nn::metric::Metric;
use parse_json::NetworkDataDe;
use std::fs;
use std::io;
use std::io::Write;
use std::time::{Duration, SystemTime};

// TODO Documentation (for everything else)
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
                .help("JSON file with input and output vectors")
        )
        .arg(
            Arg::with_name("network")
                .short('n')
                .long("network")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(true)
                .help("JSON file with network structure and hyperparameters")
        )
        .arg(
            Arg::with_name("output")
                .short('o')
                .long("output")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(false)
                .help("JSON file where weights and biases are stored (optional)")
        )
        .get_matches();

    let network_filename: &str = args.value_of("network").unwrap();
    let data_filename: &str = args.value_of("data").unwrap();

    let network_contents: String;
    let data_contents: String;

    match fs::read_to_string(&network_filename) {
        Ok(result) => network_contents = result,
        _ => return Err(format!("'{}' missing or corrupted", network_filename))
    };

    match fs::read_to_string(&data_filename) {
        Ok(result) => data_contents = result,
        _ => return Err(format!("'{}' missing or corrupted", data_filename))
    };

    match NetworkDataDe::from_json(&data_contents, &network_contents) {
        Ok(mut result) => train_from_json(&mut result, &args),
        Err(msg) => Err(msg.to_string())
    }
}

///
fn train_from_json(de_data: &mut NetworkDataDe, args: &ArgMatches) -> Result<(), String> {
    let now = SystemTime::now();
    let network: &mut Perceptron = &mut de_data.network;

    println!("Network successfully created\nStarting training cycle...\n");
    let optimizer: &mut dyn Optimizer = de_data.optimizer.as_mut();
    let metric: &dyn Metric = de_data.metric.as_ref();

    let training_set: (Array2<f64>, Array2<f64>) = (
        de_data.train_inputs.t().to_owned(),
        de_data.train_outputs.to_owned()
    );
    let validation_set: (Array2<f64>, Array2<f64>) = (
        de_data.test_inputs.t().to_owned(),
        de_data.test_outputs.to_owned()
    );

    network.fit(
        &training_set,
        &validation_set,
        optimizer,
        metric,
        de_data.cost.as_ref(),
        de_data.encoder.as_ref(),
        de_data.epochs,
    );

    let elapsed: Duration = now.elapsed().unwrap();
    println!("Finished after {} seconds\n", elapsed.as_secs_f32());

    let prediction: Array2<f64> = network.predict(&validation_set.0, de_data.encoder.as_ref());

    let metric_check: bool = metric.check(&prediction, &validation_set.1);
    let metric_value: f64 = metric.value(&prediction, &validation_set.1);

    println!("{}: {} (passed: {})\n", metric.label(), metric_check, metric_value);
    println!("Final prediction for validation set:\n{}\n", prediction);
    println!("Expected outputs from validation set:\n{}\n", &validation_set.1);

    choose_to_save(&args, &network)
}

/// 
fn choose_to_save(args: &ArgMatches, network: &Perceptron) -> Result<(), String> {
    if args.is_present("output") {
        let out_file = args.value_of("output").unwrap();
        parse_json::save_layer_values(network, out_file)
    } else {
        match user_input("\nSave internal values? (Y/N): ")
            .to_lowercase()
            .as_str()
        {
            "y" | "yes" => {
                let out_file: String = user_input("Filename: ");
                parse_json::save_layer_values(network, &out_file.as_str())
            }
            _ => Ok(())
        }
    }
}

/// 
fn user_input(prompt: &'static str) -> String {
    let mut input = String::new();
    print!("{}", prompt);

    io::stdout().flush().expect("Error: failed to flush stdout");
    io::stdin()
        .read_line(&mut input)
        .expect("Error: unable to read user input");

    input.trim().to_string()
}
