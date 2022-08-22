use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use serde_json::{Map, Value};

/// Transform outputs to/from human-readable values
pub trait Encoder {
    /// Encodes human-readable values to the same 
    /// format as the raw network output
    fn encode(&self, y: &Array2<f64>) -> Array2<f64>;

    /// Decodes the raw network output into
    /// human-readable values
    fn decode(&self, y: &Array2<f64>) -> Array2<f64>;
}

/// One-hot encoding: converts integers to 1d arrays
/// where every index is a 0 except for the index
/// corresponding to the integers value
pub struct OneHot {
    /// Maximum integer value (determines length of generated arrays)
    max: usize
}

impl OneHot {
    /// # Arguments
    ///
    /// * `params` - JSON object with initialization parameters.
    /// Allowed keys: "max"
    pub fn new(params: &Map<String, Value>) -> Self {
        let max: usize = params["max"].as_u64().unwrap_or_default() as usize;
        Self { max }
    }
}

impl Encoder for OneHot {
    fn encode(&self, y: &Array2<f64>) -> Array2<f64> {
        let row_count: usize = y.nrows();
        // Each row defaults to all zeros
        let mut one_hot: Array2<f64> = Array2::zeros((row_count, self.max + 1));
        for (mut one_hot_row, y_row) in one_hot
            .axis_iter_mut(Axis(0))
            .zip(y.axis_iter(Axis(0))) {
                // Transform integer value to index
                let el = y_row[0] as usize;
                // Corresponding index of each one-hot row becomes a one
                one_hot_row[el] = 1.0;
        }
        one_hot
    }

    fn decode(&self, y: &Array2<f64>) -> Array2<f64> {
        let y: Array2<f64> = y.t().to_owned();
        let stride: usize = y.nrows();
        let mut decoded: Vec<[f64; 1]> = vec![[0.0]; stride];
        
        for (i, row) in y.axis_iter(Axis(0)).enumerate() {
            // Get index with maximum value
            let argmax = row.argmax().unwrap() as f64;
            decoded[i] = [argmax];
        }
        Array2::from(decoded)
    }
}