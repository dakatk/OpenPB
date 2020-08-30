// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod nn;
mod parse_json;

use parse_json::NetworkDataDe;

use clap::{Arg, App};

use std::time::{SystemTime, Duration};
use std::fs;

#[doc(hidden)]
fn main() -> Result<(), String> {
    let args = App::new("Open Neural Network Benchmarker (ONNB)")
                        .version("0.1")
                        .author("Dusten Knull <dakatk97@gmail.com>")
                        .arg(Arg::with_name("data")
                            .short("d")
                            .long("data")
                            .value_name("JSON_FILE")
                            .required(true)
                        )
                        .arg(Arg::with_name("network")
                            .short("n")
                            .long("network")
                            .value_name("JSON_FILE")
                            .required(true)
                        ).get_matches();

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
            println!("\nFinished after {} seconds", elapsed.as_secs_f32());

            for (input, output) in result.inputs.iter().zip(result.outputs) {
                println!("{}: {} {}", input, network.predict(input), output);
            }

            Ok(())
        },
        Err(msg) => Err(msg.to_string())
    }
}
