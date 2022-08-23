use crate::nn::perceptron::Perceptron;
use chrono::{DateTime, Utc};
use clap::ArgMatches;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

/// Save network values to file
///
/// # Arguments
///
/// * `args` - Command line arguments
/// * `network` - Trained network to be serialized
pub fn save_to_dir(args: &ArgMatches, network: &Perceptron) -> Result<(), String> {
    let filepath: String = if args.is_present("output") {
        args.value_of("output").unwrap().to_owned()
    } else {
        let now: DateTime<Utc> = Utc::now();
        format!("output/{}.json", now.format("%d%m%y%H%M%S"))
    };
    let filepath: &Path = Path::new(filepath.as_str());

    if let Some(parent_dir) = filepath.parent() {
        match fs::create_dir_all(parent_dir) {
            Ok(_) => {}
            Err(err) => return Err(err.to_string()),
        }
    }
    save_layer_values(network, filepath)
}

/// Save internal values (weights and biases) from each layer of a network
///
/// # Arguments
///
/// * `network` - Network object to be serialized
/// * `filepath` - JSON file to write serialized values to
fn save_layer_values(network: &Perceptron, filepath: &Path) -> Result<(), String> {
    println!("\nAttempting to write to {:#?}...", filepath);

    let mut file = match File::create(filepath) {
        Ok(file) => file,
        Err(error) => {
            return Err(format!(
                "Failed to create file {:#?}: {error}",
                filepath
            ))
        }
    };

    // TODO Save values in CSV/Excel format + add other relevant data
    let network_ser = serde_json::to_string_pretty(&network).unwrap();

    match file.write_all(network_ser.as_bytes()) {
        Ok(_) => {
            println!("Success!");
            Ok(())
        }
        Err(error) => Err(error.to_string()),
    }
}
