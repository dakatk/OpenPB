use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use serde_json::Map;
use serde_json::Value;

/// 
pub trait Encoder {
    /// 
    fn encode(&self, y: &Array2<f64>) -> Array2<f64>;

    /// 
    fn decode(&self, y: &Array2<f64>) -> Array2<f64>;
}

/// 
pub struct OneHot {
    /// 
    max: usize
}

impl OneHot {
    /// # Arguments
    ///
    /// * 
    pub fn new(params: &Map<String, Value>) -> Self {
        let max: usize = params["max"].as_u64().unwrap_or_default() as usize;
        Self { max }
    }
}

impl Encoder for OneHot {
    fn encode(&self, y: &Array2<f64>) -> Array2<f64> {
        let row_count: usize = y.shape()[0];
        let mut one_hot: Array2<f64> = Array2::zeros((row_count, self.max + 1));
        for (mut one_hot_row, y_row) in one_hot
            .axis_iter_mut(Axis(0))
            .zip(y.axis_iter(Axis(0))) {
                let el = y_row[0] as usize;
                one_hot_row[el] = 1.0;
        }
        one_hot
    }

    fn decode(&self, y: &Array2<f64>) -> Array2<f64> {
        let y: Array2<f64> = y.t().to_owned();
        let stride: usize = y.shape()[0];
        let mut decoded: Vec<[f64; 1]> = vec![[0.0]; stride];
        
        for (i, row) in y.axis_iter(Axis(0)).enumerate() {
            let argmax = row.argmax().unwrap() as f64;
            decoded[i] = [argmax];
        }
        Array2::from(decoded)
    }
}