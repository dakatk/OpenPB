use ndarray::Array1;

use serde::de::{Deserialize, Deserializer, Error, Unexpected, Visitor};

use std::fmt;
use std::str;

pub struct ActivationDe {
    activationFn: Box<dyn ActivationFn>
}

impl<'de> Deserialize<'de> for ActivationDe {
    
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        struct StrVisitor;

        impl<'a> Visitor<'a> for StrVisitor {
            type Value = &'a str;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a borrowed string")
            }

            fn visit_borrowed_str<E>(self, v: &'a str) -> Result<Self::Value, E>
            where E: Error
            {   
                Ok(v)
            }

            fn visit_borrowed_bytes<E>(self, v: &'a [u8]) -> Result<Self::Value, E>
            where E: Error
            {
                str::from_utf8(v).map_err(|_| Error::invalid_value(Unexpected::Bytes(v), &self))
            }
        }

        const VARIANTS: &'static [&'static str] = &["sigmoid", "relu", "leakyrelu"];
        
        let activationName = deserializer.deserialize_str(StrVisitor)?;
        let activationFn: Box<dyn ActivationFn> = match activationName.to_lowercase().as_str() {
            "sigmoid" => Box::new(Sigmoid),
            "relu" => Box::new(ReLU),
            "leakyrelu" => Box::new(LeakyReLU),
            _ => {
                return Err(Error::unknown_variant(activationName, VARIANTS));
            }
        };

        Ok(ActivationDe { activationFn })
    }
}

/// Neuron activation function used for feed forward
/// and backprop methods in Network training
pub trait ActivationFn {

    /// Call the activation function with a set of inputs
    ///
    /// # Arguments
    ///
    /// * `x` - Row vector of input values
    fn call(&self, x: &Array1<f64>) -> Array1<f64>;

    /// First derivative of the activation function
    ///
    /// # Arguments
    ///
    /// * `x` - Row vector of input values
    fn prime(&self, x: &Array1<f64>) -> Array1<f64>;

    /// Create a clone of a boxed instance of this trait
    fn box_clone(&self) -> Box<dyn ActivationFn>;
}

/// Logistic Sigmoid activation function
#[derive(Clone)]
pub struct Sigmoid;

/// Mathematical definition of the Logistic Sigmoid function for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

/// Derivative of the Logistic Sigmoid function
///
/// # Arguments
///
/// * `x` - Function input value
fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1. - sigmoid(x))
}

impl ActivationFn for Sigmoid {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| sigmoid(el))
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| sigmoid_prime(el))
    }

    fn box_clone(&self) -> Box<dyn ActivationFn> {
        Box::new((*self).clone())
    }
}

/// Rectified Linear Unit activation function
#[derive(Clone)]
pub struct ReLU;

/// Mathematical definition of the Rectified Linear Unit 
/// function for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn relu(x: f64) -> f64 {
    if x > 0. { x } else { 0. }
}

/// Derivative of the Rectified Linear Unit function
///
/// # Arguments
///
/// * `x` - Function input value
fn relu_prime(x: f64) -> f64 {
    if x > 0. { 1. } else { 0. }
}

impl ActivationFn for ReLU {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| relu(el))
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| relu_prime(el))
    }

    fn box_clone(&self) -> Box<dyn ActivationFn> {
        Box::new((*self).clone())
    }
}

/// Leaky Rectified Linear Unit activation function
#[derive(Clone)]
pub struct LeakyReLU;

/// Mathematical definition of the Leaky ReLU
/// function for scalar values
///
/// # Arguments
///
/// * `x` - Function input value
fn leaky_relu(x: f64) -> f64 {
    if x > 0. { x } else { 0.01 * x }
}

/// Derivative of the Leaky ReLU function
///
/// # Arguments
///
/// * `x` - Function input value
fn leaky_relu_prime(x: f64) -> f64 {
    if x > 0. { 1. } else { 0.01 }
}

impl ActivationFn for LeakyReLU {
    fn call(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| leaky_relu(el))
    }

    fn prime(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|el| leaky_relu_prime(el))
    }

    fn box_clone(&self) -> Box<dyn ActivationFn> {
        Box::new((*self).clone())
    }
}