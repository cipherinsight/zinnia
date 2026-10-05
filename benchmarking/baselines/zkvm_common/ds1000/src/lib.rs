//! The DS-1000 performance-set kernels as Q32 integer Rust programs.
//!
//! Every case lives next to its `kernel.py` in
//! `benchmarking/datasets/ds1000/<case>/zkvm.rs` and is compiled into this
//! crate as the module `<case>` (see `cases.rs`). A case module provides
//!
//! * `pub type Output`: the value the guest commits as its public output,
//!   in the kernel's output order;
//! * `pub fn main(input: &mut impl Input) -> Output`: read every kernel
//!   parameter, in signature order, from the private input and compute the
//!   kernel.
//!
//! The SP1 and RISC Zero guest binaries call `main` and commit the result;
//! the hosts decode the committed bytes back into `Output` and flatten them
//! into [`Public`] values. The same `main` runs natively for the output check.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use serde::{de::DeserializeOwned, Serialize};

pub use q32::Q32;

#[cfg(feature = "std")]
pub mod fixture;
#[cfg(feature = "std")]
pub mod host;

/// Declares the case modules and the dispatch functions over them. The list
/// itself is generated into `cases.rs` by `zkvm_common/sync_cases.py`.
macro_rules! cases {
    ($($name:ident => $path:literal,)*) => {
        $(
            #[path = $path]
            pub mod $name;
        )*

        /// Every case that has a Q32 Rust program.
        pub const CASES: &[&str] = &[$(stringify!($name)),*];

        /// Decodes the committed output of `case` and flattens it into
        /// public values; `None` if the case is unknown.
        pub fn decode_publics<D: crate::Decoder>(
            case: &str,
            decoder: D,
        ) -> Option<Result<alloc::vec::Vec<crate::Public>, D::Error>> {
            use crate::Publics;
            match case {
                $(stringify!($name) => Some(decoder.decode::<$name::Output>().map(|o| o.flatten())),)*
                _ => None,
            }
        }

        /// Runs `case` on `input` and flattens its output; `None` if the
        /// case is unknown.
        pub fn run(case: &str, input: &mut impl crate::Input) -> Option<alloc::vec::Vec<crate::Public>> {
            use crate::Publics;
            match case {
                $(stringify!($name) => Some($name::main(input).flatten()),)*
                _ => None,
            }
        }
    };
}

mod cases;
pub use cases::*;

/// What every case module imports.
pub mod prelude {
    pub use crate::{Input, Shaped};
    pub use alloc::{vec, vec::Vec};
    pub use q32::{dot, int::floor_div, int::floor_mod, Q32};
}

/// The private input stream of a guest: the kernel parameters in signature
/// order, arrays flattened in row-major order.
pub trait Input {
    fn read<T: DeserializeOwned>(&mut self) -> T;

    /// A flattened array that must have exactly `len` elements.
    fn read_vec<T: DeserializeOwned>(&mut self, len: usize) -> Vec<T> {
        let values: Vec<T> = self.read();
        assert_eq!(values.len(), len, "input array has the wrong length");
        values
    }

    /// A one-dimensional array of fixed length.
    fn read_array<T: DeserializeOwned, const N: usize>(&mut self) -> [T; N] {
        match self.read_vec(N).try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("read_vec checked the length"),
        }
    }

    /// A two-dimensional array of fixed shape `R x C`.
    fn read_matrix<T: DeserializeOwned + Copy, const R: usize, const C: usize>(
        &mut self,
    ) -> [[T; C]; R] {
        let flat = self.read_vec::<T>(R * C);
        core::array::from_fn(|i| core::array::from_fn(|j| flat[i * C + j]))
    }
}

/// One public output value, as the harness compares it with NumPy, or one
/// extent of a data-dependent output shape (see [`Shaped`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Public {
    Int(i64),
    Float(Q32),
    Bool(bool),
    Dim(i64),
}

/// An array whose shape depends on the data and is not implied by its length
/// (e.g. a 2-D crop with a data-dependent number of rows and columns). The
/// guest commits the shape with the values, so the proof binds it; the host
/// reports it as `output_shape`, apart from the values compared with NumPy.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Shaped<T> {
    pub shape: Vec<i64>,
    pub values: Vec<T>,
}

impl<T: Publics> Publics for Shaped<T> {
    fn flatten_into(&self, out: &mut Vec<Public>) {
        out.extend(self.shape.iter().map(|&d| Public::Dim(d)));
        self.values.flatten_into(out);
    }
}

/// Flattening of a committed output into public values, in row-major order
/// and, for tuples, in element order.
pub trait Publics {
    fn flatten_into(&self, out: &mut Vec<Public>);

    fn flatten(&self) -> Vec<Public> {
        let mut out = Vec::new();
        self.flatten_into(&mut out);
        out
    }
}

impl Publics for Q32 {
    fn flatten_into(&self, out: &mut Vec<Public>) {
        out.push(Public::Float(*self));
    }
}

impl Publics for i64 {
    fn flatten_into(&self, out: &mut Vec<Public>) {
        out.push(Public::Int(*self));
    }
}

impl Publics for bool {
    fn flatten_into(&self, out: &mut Vec<Public>) {
        out.push(Public::Bool(*self));
    }
}

impl<T: Publics> Publics for [T] {
    fn flatten_into(&self, out: &mut Vec<Public>) {
        for item in self {
            item.flatten_into(out);
        }
    }
}

impl<T: Publics> Publics for Vec<T> {
    fn flatten_into(&self, out: &mut Vec<Public>) {
        self.as_slice().flatten_into(out);
    }
}

impl<T: Publics, const N: usize> Publics for [T; N] {
    fn flatten_into(&self, out: &mut Vec<Public>) {
        self.as_slice().flatten_into(out);
    }
}

macro_rules! tuple_publics {
    ($($name:ident),+) => {
        impl<$($name: Publics),+> Publics for ($($name,)+) {
            #[allow(non_snake_case)]
            fn flatten_into(&self, out: &mut Vec<Public>) {
                let ($($name,)+) = self;
                $($name.flatten_into(out);)+
            }
        }
    };
}
tuple_publics!(A);
tuple_publics!(A, B);
tuple_publics!(A, B, C);
tuple_publics!(A, B, C, D);

/// Turns the bytes a guest committed back into its `Output` type. Each host
/// implements it with its zkVM's serialization.
pub trait Decoder {
    type Error;
    fn decode<T: Serialize + DeserializeOwned>(self) -> Result<T, Self::Error>;
}
