// Allow boxes of specific traits to be cloned
#[doc(hidden)]
#[macro_export]
macro_rules! dyn_clone {
    ($trait_name:tt) => {
        pub trait DynClone {
            /// Create a clone of a boxed instance of a trait
            fn clone_box(&self) -> Box<dyn $trait_name>;
        }

        impl<T> DynClone for T
        where
            T: 'static + $trait_name + Clone,
        {
            fn clone_box(&self) -> Box<dyn $trait_name> {
                Box::new(self.clone())
            }
        }

        impl Clone for Box<dyn $trait_name> {
            fn clone(&self) -> Self {
                self.clone_box()
            }
        }
    };
}
