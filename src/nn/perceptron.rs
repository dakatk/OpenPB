use super::functions::activation::ActivationFn;
use super::functions::cost::Cost;
use super::functions::encoder::Encoder;
use super::functions::metric::Metric;
use super::functions::optimizer::{optimize, Optimizer};
use super::layer::Layer;
use ndarray::{Array1, Array2, ArrayViewMut1, Axis, Slice};
use rand::seq::SliceRandom;
use serde::ser::{Serialize, SerializeStruct, Serializer};
use std::fmt::Debug;

pub struct Perceptron {
    /// Input, hidden, and output layers. Each layer is considered
    /// to be 'connected' to the next one in the list
    layers: Vec<Layer>,
}

impl Perceptron {
    /// # Arguments:
    ///
    /// * `cost` - Loss function for error reporting/backprop
    pub fn new() -> Perceptron {
        Perceptron { layers: vec![] }
    }

    /// Creates a new layer and adds it to the Network. Used only for the
    /// first layer added, which is treated as the input layer
    ///
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases
    /// are present in the new Layer
    /// * `inputs` - Size of expected the Layer's input vector
    /// * `activation_fn` - Function that determines the activation of individual neurons
    fn add_input_layer(
        &mut self,
        neurons: usize,
        input_shape: (usize, usize),
        activation_fn: Box<dyn ActivationFn>,
        dropout: Option<f32>,
    ) {
        self.layers
            .push(Layer::new(neurons, input_shape, activation_fn, dropout));
    }

    /// Same as `add_input_layer`, but used for any other layer after. The number of
    /// inputs for a hidden layer is equal to the number of neurons in the preceding layer
    ///
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases
    /// are present in the new Layer
    /// * `activation_fn` - Function that determines the activation of individual neurons
    fn add_hidden_layer(
        &mut self,
        neurons: usize,
        activation_fn: Box<dyn ActivationFn>,
        dropout: Option<f32>,
    ) {
        let prev_layer: &mut Layer = self.layers.last_mut().unwrap();
        let prev_neurons: usize = prev_layer.neurons;
        let prev_inputs: usize = prev_layer.inputs.ncols();

        self.layers.push(Layer::new(
            neurons,
            (prev_neurons, prev_inputs),
            activation_fn,
            dropout,
        ));
    }

    /// Add a Layer to the next open spot in the Network's structure. This function
    /// also dynamically expands the Network's overall size
    ///
    /// # Arguments
    ///
    /// * `neurons` - Number of neurons, determines how many weights/biases
    /// are present in the new Layer
    /// * `inputs` (optional) - Size of expected the Layer's input vector
    /// * `activation_fn` - Function that determines the activation of individual neurons
    pub fn add_layer(
        &mut self,
        neurons: usize,
        input_shape: Option<(usize, usize)>,
        activation_fn: Box<dyn ActivationFn>,
        dropout: Option<f32>,
    ) {
        match input_shape {
            Some(input_shape) => self.add_input_layer(neurons, input_shape, activation_fn, dropout),
            _ => self.add_hidden_layer(neurons, activation_fn, dropout),
        }
    }

    /// Trains the entire Network for a specified number of cycles. Training is
    /// stopped when the given metric is satisfied based on the input/output
    /// sets provided
    ///
    /// # Arguments
    ///
    /// * `training_set` - Set of all input and output vectors to train the network on
    /// * `validation_set` - Set of all input and output vectors to validate if the
    /// network has been sufficiently trained
    /// * `optimizer` - Optimization method used when performing gradient descent
    /// * `metric` - Decides when the Network is performing 'good enough'
    /// on the provided validation data
    /// * `cost` -
    /// * `encoder` -
    /// * `epochs` - Maximum number of training cycles
    /// * `shuffle` - When 'true', training inputs are shuffled at the start of
    /// each training cycle
    ///
    /// # Returns
    ///
    /// The number of epochs it took for the training to complete (metric check passed)
    pub fn fit(
        &mut self,
        training_set: &(Array2<f64>, Array2<f64>),
        validation_set: &(Array2<f64>, Array2<f64>),
        optimizer: &mut dyn Optimizer,
        metric: &dyn Metric,
        cost: &dyn Cost,
        encoder: &dyn Encoder,
        epochs: usize,
        shuffle: bool,
        batch_size: Option<usize>,
    ) -> usize {
        // Keep track of which iteration training ended on
        // (default is the maximum number of epochs)
        let mut last_epoch: usize = epochs;

        // Rows and columns of full training input set
        let input_rows: usize = training_set.0.nrows();
        let input_cols: usize = training_set.0.ncols();

        // Split training set
        let mut training_inputs: Array2<f64> = training_set.0.clone();
        let mut training_outputs: Array2<f64> = training_set.1.clone();

        // Split validation set
        let validation_inputs: &Array2<f64> = &validation_set.0;
        let validation_outputs: &Array2<f64> = &validation_set.1;

        // Encode training set output values to match
        // the network's output format
        let mut expected: Array2<f64> = encoder.encode(&training_outputs).t().to_owned();

        // Initiate RNG
        let mut rng = rand::thread_rng();

        // Starting index of batch, if applicable
        let mut batch_start: usize = 0;

        for epoch in 1..=epochs {
            if shuffle {
                // Assumes each input vector has a single corresponding output vector
                // (number of columns of the training inputs should be
                // equal to the number of rows of the outputs after transposing)
                let mut indices: Vec<usize> = (0..training_inputs.ncols()).collect();
                indices.shuffle(&mut rng);

                self.shuffle_on_axis(&mut training_inputs, &indices, Axis(1));
                self.shuffle_on_axis(&mut training_outputs, &indices, Axis(0));
            }

            if let Some(batch_size) = batch_size {
                // Create minibatches by slicing training sets
                training_inputs = self.batch(&training_set.0, batch_start, batch_size, Axis(1));
                training_outputs = self.batch(&training_set.1, batch_start, batch_size, Axis(0));

                // Re-evaluate expected values for minibatch
                expected = encoder.encode(&training_outputs).t().to_owned();

                // Increment batch start index
                batch_start += batch_size;
                if batch_start > input_cols {
                    batch_start = 0;
                }
            }
            // Check network prediction against validation set
            let prediction: Array2<f64> = self.predict(validation_inputs, encoder);
            let early_stop: bool = metric.check(&prediction, validation_outputs);

            // Stop training if early stopping metric criteria has been met
            if early_stop {
                last_epoch = epoch;
                break;
            }

            let actual: Array2<f64> = self.feed_forward(&training_inputs);
            let delta: Array2<f64> = cost.prime(&actual, &expected);
            self.back_prop(&delta);

            // Update network weights/biases using
            // the given Optimizer
            optimize(optimizer, &mut self.layers, input_rows);
        }
        last_epoch
    }

    /// Shuffle matrix rows or cols in-place
    ///
    /// # Arguments
    ///
    /// * `values` - Matrix to be shuffled
    /// * `indices` - Generated list of shuffled indices along given axis
    /// * `axis` - Axis in which vectors are shuffled
    fn shuffle_on_axis(&self, values: &mut Array2<f64>, indices: &Vec<usize>, axis: Axis) {
        let new_rows: Vec<Array1<f64>> = indices
            .iter()
            .map(|index| values.index_axis(axis, *index).to_owned())
            .collect();

        for (i, new_row) in new_rows.iter().enumerate() {
            let mut row: ArrayViewMut1<f64> = values.index_axis_mut(axis, i);
            row.assign(new_row);
        }
    }

    fn batch(
        &self,
        values: &Array2<f64>,
        start: usize,
        batch_size: usize,
        axis: Axis,
    ) -> Array2<f64> {
        let end: usize = start + batch_size;
        let end = end.min(values.len_of(axis));
        let indices: Slice = Slice::from(start..end);

        values.slice_axis(axis, indices).to_owned()
    }

    /// Performs the feedforward step for all Layers to return the
    /// network's prediction for a given input vector
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix of input vectors
    pub fn feed_forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut output: Array2<f64> = inputs.to_owned();
        for layer in self.layers.iter_mut() {
            output = layer.feed_forward(&output);
        }
        output
    }

    /// Performs the backpropogation step for all layers to calculate
    /// the appropriate deltas for the optimization step
    ///
    /// # Arguments
    ///
    /// * `deltas` - Delta values matrix calculated from output layer
    pub fn back_prop(&mut self, deltas: &Array2<f64>) {
        let mut attached_layer: Option<&Layer> = None;
        for layer in self.layers.iter_mut().rev() {
            match attached_layer {
                Some(attached_layer) => layer.back_prop(attached_layer),
                None => layer.back_prop_with_deltas(deltas),
            };
            attached_layer = Some(layer);
        }
    }

    /// Computes the network's prediction for a given input.
    /// Assumes the network has already been trained, therefore
    /// Dropout Regularization is not taken into account
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix of input vectors
    /// * `encoder` - Method for decoding output to readable values
    pub fn predict(&mut self, inputs: &Array2<f64>, encoder: &dyn Encoder) -> Array2<f64> {
        let mut prev_outputs: Array2<f64> = inputs.to_owned();
        for layer in self.layers.iter_mut() {
            prev_outputs = layer.predict(&prev_outputs);
        }
        encoder.decode(&prev_outputs)
    }
}

impl Serialize for Perceptron {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("Perceptron", 1)?;
        s.serialize_field("layers", &self.layers)?;
        s.end()
    }
}

impl Debug for Perceptron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Only returns number of layers, not the information contained
        // within each layer
        f.debug_struct("Perceptron")
            .field("layers", &self.layers.len())
            .finish()
    }
}
