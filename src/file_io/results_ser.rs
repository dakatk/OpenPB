use crate::nn::perceptron::Perceptron;
use ndarray::Array2;
use serde::Serialize;

/// Serialized data for the metric that
/// was used during training
#[derive(Serialize, Debug)]
struct MetricSer {
    /// Name (label) of the metric
    name: String,
    /// Metric value (score) for the network's
    /// prediction after training has concluded
    value: f64,
    /// Whether or not the the metric's score
    /// is considered a "passing" score
    passed: bool,
}

#[derive(Serialize, Debug)]
pub struct TrainingResultsSer {
    /// Trained network
    network: Perceptron,
    /// Data for the metric that was used to
    /// validate the network's results during training
    metric: MetricSer,
    /// Time it took for training to complete
    /// (in seconds)
    elapsed_time: f32,
    /// Total number of iterations until the
    /// network was considered fully trained
    total_epochs: usize,
    /// Predicted values from feeding validtion
    /// set inputs into the trained network
    predicted_output: Array2<f64>,
}

impl TrainingResultsSer {
    pub fn new(
        network: Perceptron,
        metric_label: String,
        metric_value: f64,
        metric_passed: bool,
        elapsed_time: f32,
        total_epochs: usize,
        predicted_output: Array2<f64>,
    ) -> Self {
        let metric: MetricSer = MetricSer {
            name: metric_label,
            value: metric_value,
            passed: metric_passed,
        };
        Self {
            network,
            metric,
            elapsed_time,
            total_epochs,
            predicted_output,
        }
    }
}

#[derive(Serialize, Debug)]
pub struct ThreadedResultsSer {
    /// Collection of serialized training
    /// results from each thread
    all_results: Vec<TrainingResultsSer>,
    /// Input values used when validating
    /// the network
    validation_inputs: Array2<f64>,
    /// Output values to validate the
    /// network against
    validation_outputs: Array2<f64>,
    /// Size of minibatches (if applicable)
    batch_size: Option<usize>,
}

impl ThreadedResultsSer {
    pub fn new(
        all_results: Vec<TrainingResultsSer>,
        validation_inputs: Array2<f64>,
        validation_outputs: Array2<f64>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            all_results,
            validation_inputs,
            validation_outputs,
            batch_size,
        }
    }
}
