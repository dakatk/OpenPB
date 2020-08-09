mod nn;

use nn::activations::Sigmoid;
use nn::costs::MSE;
use nn::network::Network;
use nn::optimizers::SGD;

use ndarray::{Array, Array1};

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
    let optimizer = SGD::new(0.9, 0.1);
    let cost = MSE {};

    let network = Network::new(Box::new(optimizer), Box::new(cost));
    network.add_layer(8, Some(2), Box::new(Sigmoid {}));
    network.add_layer(1, None, Box::new(Sigmoid {}));

    network.fit(inputs, outputs, 1000);

    for (input, output) in inputs.iter().zip(outputs) {
        print!("{}: {} {}", input, network.predict(input), output);
    }
}
