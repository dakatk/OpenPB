use crate::file_io::parse_json::NetworkDataDe;
use crate::file_io::results_ser::{TrainingResultsSer, ThreadedResultsSer};
use crate::file_io::save_output;
use crate::nn::functions::cost::Cost;
use crate::nn::functions::encoder::Encoder;
use crate::nn::functions::metric::Metric;
use crate::nn::functions::optimizer::Optimizer;
use crate::nn::perceptron::Perceptron;
use clap::ArgMatches;
use ndarray::Array2;
use std::thread::{self, JoinHandle};
use std::time::SystemTime;
use std::usize;

const DEFAULT_THREADS: usize = 1;

/// Train network with deserailzed JSON data
/// 
/// # Arguments
/// 
/// * `de_data` - 
/// * `arg` - 
pub fn train_from_json(de_data: &NetworkDataDe, args: &ArgMatches) -> Result<(), String> {
    let total_threads: usize = *args.get_one("threads").unwrap_or(&DEFAULT_THREADS);
    let shuffle: bool = *args.get_one("shuffle").unwrap_or(&false);

    let mut training_threads: Vec<JoinHandle<TrainingResultsSer>> = vec![];
    let mut all_results: Vec<TrainingResultsSer> = vec![];

    // Create training threads
    for id in 0..total_threads {
        // TODO The use of 'clone()' here is inefficient...
        training_threads.push(train_single_thread(id, de_data.clone(), shuffle));
    }

    // Wait for each training thread to finish, then add the data
    // to a Vec containing all training results
    for thread in training_threads {
        all_results.push(thread.join().unwrap());
    }

    // Isolate validation inputs
    let validation_inputs: Array2<f64> = de_data.test_inputs.t().to_owned();
    // Isolate validation outputs
    let validation_outputs: Array2<f64> = de_data.test_outputs.to_owned();

    let threaded_results = ThreadedResultsSer::new(
        all_results,
        validation_inputs,
        validation_outputs
    );
    save_output::save_to_dir(&args, threaded_results)
}

/// 
fn train_single_thread(
    id: usize, 
    mut de_data: NetworkDataDe,
    shuffle: bool
) -> JoinHandle<TrainingResultsSer> {
    thread::spawn(move || {
        // Create new network with randomized weights and biases
        let mut network: Perceptron = de_data.create_network().unwrap();

        // Get dyn references from boxed traits
        let optimizer: &mut dyn Optimizer = de_data.optimizer.as_mut();
        let metric: &dyn Metric = de_data.metric.as_ref();
        let cost: &dyn Cost = de_data.cost.as_ref();
        let encoder: &dyn Encoder = de_data.encoder.as_ref();

        // Isolate training set
        let training_set: (Array2<f64>, Array2<f64>) = (
            de_data.train_inputs.t().to_owned(),
            de_data.train_outputs.to_owned(),
        );
        // Isolate validation set
        let validation_set: (Array2<f64>, Array2<f64>) = (
            de_data.test_inputs.t().to_owned(),
            de_data.test_outputs.to_owned(),
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
            de_data.epochs,
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