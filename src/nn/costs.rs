use ndarray::Array1;

/// Cost or loss function to determine the Network's error
pub trait Cost {
    /// First derivative of the cost function. Used in Network backprop
    ///
    /// # Arguments
    ///
    /// `o` - Actual values
    /// `y` - Expected values
    fn prime(&self, o: &Array1<f64>, y: &Array1<f64>) -> Array1<f64>;
    /// Clones this trait when boxed in lieu of copying
    fn box_clone(&self) -> Box<dyn Cost>;
}

/// Mean Squared Error loss function
#[derive(Clone)]
pub struct MSE;

impl Cost for MSE {
    fn prime(&self, o: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
        o - y
    }

    fn box_clone(&self) -> Box<dyn Cost> {
        Box::new((*self).clone())
    }
}
