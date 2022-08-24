// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod file_io;
mod nn;
mod trainer;

use file_io::parse_json::NetworkDataDe;
use trainer::train_from_json;
use clap::{value_parser, App, Arg, ArgMatches};
use std::fs;

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
                .help("JSON file with training and validation sets (required)"),
        )
        .arg(
            Arg::with_name("network")
                .short('n')
                .long("network")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(true)
                .help("JSON file with network structure and hyperparameters (required)"),
        )
        .arg(
            Arg::with_name("output")
                .short('o')
                .long("output")
                .takes_value(true)
                .value_name("JSON_FILE")
                .required(false)
                .help("JSON file where training results are stored (optional)"),
        )
        .arg(
            Arg::with_name("threads")
                .short('t')
                .long("threads")
                .value_parser(value_parser!(usize))
                .takes_value(true)
                .value_name("NUMBER_OF_THREADS")
                .required(false)
                .help("Number of threads spawned to train multiple smaple of the same network setup (optional)")
        )
        .arg(
            Arg::with_name("shuffle")
                .short('s')
                .long("shuffle")
                .value_parser(value_parser!(bool))
                .takes_value(true)
                .value_name("SHUFFLE")
                .required(false)
                .help("Flag that indicates whether or not to shuffle training data during each cycle (optional)")
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
        Ok(mut de_data) => train_from_json(&mut de_data, &args),
        Err(error) => Err(error.to_string()),
    }
}
