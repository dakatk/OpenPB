use crate::file_io::parse_json::NetworkDataDe;
use crate::file_io::results_ser::{TrainingResultsSer, ThreadedResultsSer};
use crate::file_io::save_output;
use crate::nn::functions::cost::Cost;
use crate::nn::functions::encoder::Encoder;
use crate::nn::functions::metric::Metric;
use crate::nn::functions::optimizer::Optimizer;
use crate::nn::perceptron::Perceptron;
use crate::args::Args;
use ndarray::Array2;
use std::thread::{self, JoinHandle};
use std::time::SystemTime;
use std::usize;

/// Train network with deserailzed JSON data
/// 
/// # Arguments
/// 
/// * `network_data_de` - Deserialized network parameters with
/// training and validation data 
/// * `args` - Command line arguments
pub fn train_from_json(network_data_de: NetworkDataDe, args: Args) -> Result<(), String> {
    let mut training_threads: Vec<JoinHandle<TrainingResultsSer>> = vec![];
    let mut all_results: Vec<TrainingResultsSer> = vec![];

    // Create training threads
    for id in 0..args.threads {
        training_threads.push(train_single_thread(id, network_data_de.clone(), args.shuffle));
    }

    // Wait for each training thread to finish, then add the data
    // to a Vec containing all training results
    for thread in training_threads {
        all_results.push(thread.join().unwrap());
    }

    // Isolate validation inputs
    let validation_inputs: Array2<f64> = network_data_de.test_inputs.t().to_owned();
    // Isolate validation outputs
    let validation_outputs: Array2<f64> = network_data_de.test_outputs.to_owned();

    let threaded_results = ThreadedResultsSer::new(
        all_results,
        validation_inputs,
        validation_outputs
    );
    save_output::save_to_dir(args, threaded_results)
}

/// Create new training thread
fn train_single_thread(
    id: usize, 
    mut network_data_de: NetworkDataDe,
    shuffle: bool
) -> JoinHandle<TrainingResultsSer> {
    thread::spawn(move || {
        // Create new network with randomized weights and biases
        let mut network: Perceptron = network_data_de.create_network().unwrap();

        // Get dyn references from boxed traits
        let optimizer: &mut dyn Optimizer = network_data_de.optimizer.as_mut();
        let metric: &dyn Metric = network_data_de.metric.as_ref();
        let cost: &dyn Cost = network_data_de.cost.as_ref();
        let encoder: &dyn Encoder = network_data_de.encoder.as_ref();

        // Isolate training set
        let training_set: (Array2<f64>, Array2<f64>) = (
            network_data_de.train_inputs.t().to_owned(),
            network_data_de.train_outputs.to_owned(),
        );
        // Isolate validation set
        let validation_set: (Array2<f64>, Array2<f64>) = (
            network_data_de.test_inputs.t().to_owned(),
            network_data_de.test_outputs.to_owned(),
        );

        // Start time before training begins
        let now: SystemTime = SystemTime::now();

        println!("Network initialized, starting training cycle for thread {id}...");
        let total_epochs: u64 = network.fit(
            &training_set,
            &validation_set,
            optimizer,
            metric,
            cost,
            encoder,
            network_data_de.epochs,
            shuffle,
        );
        println!("Training finished for thread {id}!");

        let validation_inputs: &Array2<f64> = &validation_set.0;
        let validation_outputs: &Array2<f64> = &validation_set.1;

        // Total time after training finished
        let elapsed_time: f32 = now.elapsed().unwrap().as_secs_f32();
        // Prediction from feeding validation inputs into trained network 
        let predicted_output: Array2<f64> = network.predict(validation_inputs, encoder);

        // Metric values
        let metric_label: String = metric.label().to_string();
        let metric_value: f64 = metric.value(&predicted_output, validation_outputs);
        let metric_passed: bool = metric.check(&predicted_output, validation_outputs);

        TrainingResultsSer::new(
            network,
            metric_label,
            metric_value,
            metric_passed,
            elapsed_time,
            total_epochs,
            predicted_output
        )
    })
}