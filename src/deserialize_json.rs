mod nn;

use nn::activations::{ActivationFn, Sigmoid, ReLU, LeakyReLU};

use serde::de::{Deserialize, Deserializer, Error, SeqAccess, Visitor};

use std::fmt;

struct LayerDe {

    neurons: u64,
    activation_fn: Box<dyn ActivationFn>
}

impl<'de> Deserialize<'de> for LayerDe {
    
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {

        struct LayerVisitor;

        impl<'de> Visitor<'de> for LayerVisitor {
            type Value = LayerDe;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct LayerDe")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error> 
            where A: SeqAccess<'de> {
                let neurons: u64 = match seq.next_element()? {
                    Some(value) => value,
                    None => {
                        return Err(Error::invalid_length(0, &self));
                    }
                };

                let activation_name: &str = match seq.next_element()? {
                    Some(value) => value,
                    None => {
                        return Err(Error::invalid_length(1, &self));
                    }
                };

                let activation_fn: Box<dyn ActivationFn> = match activation_name.to_lowercase().as_str() {
                    "sigmoid" => Box::new(Sigmoid),
                    "relu" => Box::new(ReLU),
                    "leakyrelu" => Box::new(LeakyReLU),
                    _ => {
                        return Err(Error::custom(format!("Invalid activation function name: {}", activation_name)));
                    }
                };

                Ok(LayerDe {
                    neurons,
                    activation_fn
                })
            } 
        }

        const FIELDS: &'static [&'static str] = &["neurons", "activation"];
        deserializer.deserialize_struct("LayerDe", FIELDS, LayerVisitor)
    }
}