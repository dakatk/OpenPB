use clap::Parser;

#[doc(hidden)]
#[derive(Parser, Debug)]
#[clap(author = "Dusten Knull <dakatk97@gmail.com>", version = "0.1", name = "Open Neural Network Benchmarker (ONNB)", about, long_about = None)]
pub struct Args {
    /// JSON file with training and validation sets (required)
    #[clap(short, long, value_parser)]
    pub data: String,
    /// JSON file with network structure and hyperparameters (required)
    #[clap(short, long, value_parser)]
    pub network: String,
    /// JSON file where training results are stored (optional)
    #[clap(short, long, value_parser)]
    pub output: Option<String>,
    /// Number of threads spawned to train multiple samples of the same network setup (optional)
    #[clap(short, long, value_parser, default_value_t = 1)]
    pub threads: usize,
    /// Flag that indicates whether or not to shuffle training data during each cycle (optional)
    #[clap(short, long, value_parser, default_value_t = false)]
    pub shuffle: bool,
    /// Maximum number of epochs (iterations) until training loop finishes (required)
    #[clap(short, long, value_parser)]
    pub epochs: usize,
    /// Maximum number of input vectors trained during each cycle (optional)
    #[clap(short, long, value_parser)]
    pub batch_size: Option<usize>, 
}
