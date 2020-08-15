// To generate docs for this project, run command:
// cargo doc --open --no-deps --document-private-items
mod nn;

use nn::activations::Sigmoid;
use nn::costs::MSE;
use nn::metrics::Accuracy;
use nn::network::Network;
use nn::optimizers::SGD;

use ndarray::{Array, Array1};

#[doc(hidden)]
fn main() {
    let inputs: Vec<Array1<f64>> = vec![
        Array::from(vec![0.0, 0.0]),
        Array::from(vec![0.0, 1.0]),
        Array::from(vec![1.0, 0.0]),
        Array::from(vec![1.0, 1.0]),
    ];
    let outputs: Vec<Array1<f64>> = vec![
        Array::from(vec![0.0]),
        Array::from(vec![1.0]),
        Array::from(vec![1.0]),
        Array::from(vec![0.0]),
    ];
    let cost = MSE {};

    let mut network = Network::new(Box::new(cost));
    network.add_layer(8, Some(2), Box::new(Sigmoid {}));
    network.add_layer(1, None, Box::new(Sigmoid {}));

    let optimizer = SGD::new(0.01);
    let metric = Accuracy::new(1);

    network.fit(
        &inputs,
        &outputs,
        Box::new(optimizer),
        Box::new(metric),
        10000,
    );

    for (input, output) in inputs.iter().zip(outputs) {
        println!("{}: {} {}", input, network.predict(input), output);
    }
}
