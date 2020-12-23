// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod nn;
mod parse_json;

use nn::network::Network;

use parse_json::NetworkDataDe;

use clap::{App, Arg, ArgMatches};

use std::fs;
use std::io;
use std::io::Write;
use std::time::{Duration, SystemTime};

#[doc(hidden)]
fn main() -> Result<(), String> {
    // TODO extract this to YAML file?
    let args: ArgMatches = App::new("Open Neural Network Benchmarker (ONNB)")
        .version("0.1")
        .author("Dusten Knull <dakatk97@gmail.com>")
        .arg(
            Arg::with_name("data")
                .short("d")
                .long("data")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(true)
                .help("JSON file with input and output vectors")
        )
        .arg(
            Arg::with_name("network")
                .short("n")
                .long("network")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(true)
                .help("JSON file with network structure and hyperparameters")
        )
        .arg(
            Arg::with_name("output")
                .short("o")
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
        Ok(result) => {
            let now = SystemTime::now();
            let mut network = result.network;

            println!("Network successfully created\nStarting training cycle...\n");

            network.fit(
                &result.inputs,
                &result.outputs,
                result.optimizer,
                result.metric,
                result.epochs
            );

            let elapsed: Duration = now.elapsed().unwrap();
            println!("Finished after {} seconds\n", elapsed.as_secs_f32());

            for (input, output) in result.inputs.iter().zip(result.outputs) {
                println!("{}: {} {}", input.t(), network.predict(input).t(), output.t());
            }

            choose_to_save(&args, &network)
        }
        Err(msg) => Err(msg.to_string())
    }
}

fn choose_to_save(args: &ArgMatches, network: &Network) -> Result<(), String> {
    if args.is_present("output") {
        let out_file = args.value_of("output").unwrap();
        parse_json::save_layer_values(network, out_file)
    } else {
        match user_input("\nSave internal values? (Y/N): ")
            .to_lowercase()
            .as_str()
        {
            "y" | "yes" => {
                let out_file = user_input("Filename: ");
                parse_json::save_layer_values(network, &out_file.as_str())
            }
            _ => Ok(())
        }
    }
}

fn user_input(prompt: &'static str) -> String {
    let mut input = String::new();
    print!("{}", prompt);

    io::stdout().flush().expect("Error: failed to flush stdout");
    io::stdin()
        .read_line(&mut input)
        .expect("Error: unable to read user input");

    input.trim().to_string()
}
