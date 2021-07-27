use ndarray::Array2;

/// Cost or loss function to determine the Network's error
pub trait Cost: DynClone {
    /// First derivative of the cost function. Used in Network backprop
    ///
    /// # Arguments
    ///
    /// * `o` - Actual values
    /// * `y` - Expected values
    fn prime(&self, o: &Array2<f64>, y: &Array2<f64>) -> Array2<f64>;
}

pub trait DynClone {
    /// Create a clone of a boxed instance of a trait
    fn clone_box(&self) -> Box<dyn Cost>;
}

impl<T> DynClone for T
where
    T: 'static + Cost + Clone
{
    fn clone_box(&self) -> Box<dyn Cost> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Cost> {
    fn clone(&self) -> Box<dyn Cost> {
        self.clone_box()
    }
}

/// Mean Squared Error loss function
#[derive(Clone)]
pub struct MSE;

impl Cost for MSE {
    fn prime(&self, o: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        2.0 * (o - y)
    }
}
