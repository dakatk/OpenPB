// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod args;
mod file_io;
mod nn;
mod trainer;

use args::Args;
use clap::Parser;
use file_io::json_de::NetworkDataDe;
use std::fs;
use trainer::train_from_json;

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
        Err(error) => Err(error),
    }
}
