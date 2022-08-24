// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod args;
mod trainer;
mod nn;
mod file_io;

use clap::Parser;
use file_io::parse_json::NetworkDataDe;
use trainer::train_from_json;
use args::Args;
use std::fs;

#[doc(hidden)]
fn main() -> Result<(), String> {
    let args = Args::parse();

    let network_json: String = match fs::read_to_string(&args.network) {
        Ok(result) => result,
        _ => return Err(format!("File {} missing or corrupted", args.network)),
    };
    let data_json: String = match fs::read_to_string(&args.data) {
        Ok(result) => result,
        _ => return Err(format!("File {} missing or corrupted", args.data)),
    };

    match NetworkDataDe::from_json(&data_json, &network_json) {
        Ok(network_data_de) => train_from_json(network_data_de, args),
        Err(error) => Err(error.to_string()),
    }
}
