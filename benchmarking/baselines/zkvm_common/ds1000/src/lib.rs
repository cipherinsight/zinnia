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
//! the hosts decode the committed words back into `Output` and flatten them
//! into [`Public`] values. The same `main` runs natively for the output check.
//!
//! **Encoding.** Inputs and outputs travel as 64-bit words, through the
//! zkVMs' raw-slice I/O (one copy per buffer) rather than their serde codecs,
//! which cost hundreds of cycles per element (measured on RISC Zero: 287
//! cycles per element to read an array with `env::read`, 202 to commit it
//! with `env::commit`). A parameter is one buffer of words (a scalar is one
//! word); the output is one buffer, laid out by [`Publics::encode`].
//!
//! A case imports [`prelude`]; NumPy behaviours that several cases share
//! live in [`np`].
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

pub use q32::Q32;

#[cfg(feature = "std")]
pub mod fixture;
#[cfg(feature = "std")]
pub mod host;
pub mod np;

/// Declares the case modules and the dispatch functions over them. The list
/// itself is generated into `cases.rs` by `zkvm_common/sync_cases.py`.
macro_rules! cases {
    ($($name:ident => $path:literal $(, unused [$($unused:literal),*])?;)*) => {
        $(
            #[path = $path]
            pub mod $name;
        )*

        /// Every case that has a Q32 Rust program.
        pub const CASES: &[&str] = &[$(stringify!($name)),*];

        /// Kernel parameters that `case` never reads, declared in its
        /// program by a line `// unused parameters: <names>`. Zinnia removes
        /// such inputs from the circuit, so the guest does not take them
        /// either: the hosts leave them out of the private input.
        pub fn unused_params(case: &str) -> &'static [&'static str] {
            match case {
                $(stringify!($name) => &[$($($unused),*)?],)*
                _ => &[],
            }
        }

        /// Decodes the words `case` committed and flattens them into public
        /// values; `None` if the case is unknown.
        pub fn decode_publics(
            case: &str,
            words: &[i64],
        ) -> Option<Result<alloc::vec::Vec<crate::Public>, crate::DecodeError>> {
            match case {
                $(stringify!($name) => Some(crate::decode_all::<$name::Output>(words).map(|o| crate::Publics::flatten(&o))),)*
                _ => None,
            }
        }

        /// Runs `case` on `input` and returns the words it commits; `None`
        /// if the case is unknown.
        pub fn run(case: &str, input: &mut impl crate::Input) -> Option<alloc::vec::Vec<i64>> {
            match case {
                $(stringify!($name) => Some(crate::Publics::encode_to_vec(&$name::main(input))),)*
                _ => None,
            }
        }
    };
}

mod cases;
pub use cases::*;

/// What every case module imports.
pub mod prelude {
    pub use crate::np::{self, np_index};
    pub use crate::{Input, Shaped};
    pub use alloc::{vec, vec::Vec};
    pub use q32::{dot, int::floor_div, int::floor_mod, Q32};
}

/// A value that travels as one 64-bit word: a Q32 by its encoding, an
/// integer as itself, a boolean as 0 or 1.
pub trait Word: Copy {
    fn to_word(self) -> i64;
    /// `None` if `word` encodes no value of this type.
    fn from_word(word: i64) -> Option<Self>;
}

impl Word for Q32 {
    fn to_word(self) -> i64 {
        self.to_bits()
    }
    fn from_word(word: i64) -> Option<Q32> {
        Some(Q32::from_bits(word))
    }
}

impl Word for i64 {
    fn to_word(self) -> i64 {
        self
    }
    fn from_word(word: i64) -> Option<i64> {
        Some(word)
    }
}

impl Word for bool {
    fn to_word(self) -> i64 {
        self as i64
    }
    fn from_word(word: i64) -> Option<bool> {
        match word {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

/// The private input stream of a guest: the kernel parameters in signature
/// order, each one buffer of words, arrays flattened in row-major order.
pub trait Input {
    /// The next parameter's words.
    fn read_words(&mut self) -> Vec<i64>;

    /// A scalar parameter.
    fn read<T: Word>(&mut self) -> T {
        let words = self.read_words();
        assert_eq!(words.len(), 1, "a scalar parameter is one word");
        T::from_word(words[0]).expect("parameter has the wrong type")
    }

    /// A flattened array that must have exactly `len` elements.
    fn read_vec<T: Word>(&mut self, len: usize) -> Vec<T> {
        let words = self.read_words();
        assert_eq!(words.len(), len, "input array has the wrong length");
        words.into_iter().map(|w| T::from_word(w).expect("array has the wrong element type")).collect()
    }

    /// A one-dimensional array of fixed length.
    fn read_array<T: Word, const N: usize>(&mut self) -> [T; N] {
        match self.read_vec(N).try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("read_vec checked the length"),
        }
    }

    /// A two-dimensional array of fixed shape `R x C`.
    fn read_matrix<T: Word, const R: usize, const C: usize>(&mut self) -> [[T; C]; R] {
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
#[derive(Clone, Debug, PartialEq)]
pub struct Shaped<T> {
    pub shape: Vec<i64>,
    pub values: Vec<T>,
}

/// The committed words do not describe a value of the case's output type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeError;

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("the committed words do not match the output type")
    }
}

impl core::error::Error for DecodeError {}

/// A value a guest makes public: how it is committed as 64-bit words (a
/// scalar is one word, a vector its length then its items, an array or a
/// tuple its items in order), decoded again by the host, and flattened into
/// public values in row-major order.
pub trait Publics: Sized {
    fn encode(&self, out: &mut Vec<i64>);
    fn decode(words: &mut core::slice::Iter<'_, i64>) -> Result<Self, DecodeError>;
    fn flatten_into(&self, out: &mut Vec<Public>);

    fn encode_to_vec(&self) -> Vec<i64> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    fn flatten(&self) -> Vec<Public> {
        let mut out = Vec::new();
        self.flatten_into(&mut out);
        out
    }
}

/// Decodes `words` as one `T` and requires that nothing is left over.
pub fn decode_all<T: Publics>(words: &[i64]) -> Result<T, DecodeError> {
    let mut iter = words.iter();
    let value = T::decode(&mut iter)?;
    if iter.next().is_some() {
        return Err(DecodeError);
    }
    Ok(value)
}

fn next_word<T: Word>(words: &mut core::slice::Iter<'_, i64>) -> Result<T, DecodeError> {
    words.next().and_then(|&w| T::from_word(w)).ok_or(DecodeError)
}

macro_rules! scalar_publics {
    ($($ty:ty => $variant:ident),*) => {
        $(
            impl Publics for $ty {
                fn encode(&self, out: &mut Vec<i64>) {
                    out.push(self.to_word());
                }
                fn decode(words: &mut core::slice::Iter<'_, i64>) -> Result<Self, DecodeError> {
                    next_word(words)
                }
                fn flatten_into(&self, out: &mut Vec<Public>) {
                    out.push(Public::$variant(*self));
                }
            }
        )*
    };
}
scalar_publics!(Q32 => Float, i64 => Int, bool => Bool);

impl<T: Publics> Publics for Vec<T> {
    fn encode(&self, out: &mut Vec<i64>) {
        out.push(self.len() as i64);
        for item in self {
            item.encode(out);
        }
    }
    fn decode(words: &mut core::slice::Iter<'_, i64>) -> Result<Self, DecodeError> {
        let len: i64 = next_word(words)?;
        let len = usize::try_from(len).map_err(|_| DecodeError)?;
        if len > words.len() {
            return Err(DecodeError);
        }
        (0..len).map(|_| T::decode(words)).collect()
    }
    fn flatten_into(&self, out: &mut Vec<Public>) {
        for item in self {
            item.flatten_into(out);
        }
    }
}

impl<T: Publics> Publics for Shaped<T> {
    fn encode(&self, out: &mut Vec<i64>) {
        self.shape.encode(out);
        self.values.encode(out);
    }
    fn decode(words: &mut core::slice::Iter<'_, i64>) -> Result<Self, DecodeError> {
        Ok(Shaped { shape: Vec::decode(words)?, values: Vec::decode(words)? })
    }
    fn flatten_into(&self, out: &mut Vec<Public>) {
        out.extend(self.shape.iter().map(|&d| Public::Dim(d)));
        self.values.flatten_into(out);
    }
}

macro_rules! tuple_publics {
    ($($name:ident),+) => {
        impl<$($name: Publics),+> Publics for ($($name,)+) {
            #[allow(non_snake_case)]
            fn encode(&self, out: &mut Vec<i64>) {
                let ($($name,)+) = self;
                $($name.encode(out);)+
            }
            fn decode(words: &mut core::slice::Iter<'_, i64>) -> Result<Self, DecodeError> {
                Ok(($($name::decode(words)?,)+))
            }
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

// The byte views below rely on the in-memory layout of `i64` being its
// little-endian encoding, as on x86-64 and RISC-V.
const _: () = assert!(cfg!(target_endian = "little"), "words travel as little-endian bytes");

/// The little-endian bytes of `words`, without copying, for zkVM APIs that
/// take bytes.
pub fn words_as_le_bytes(words: &[i64]) -> &[u8] {
    bytemuck::cast_slice(words)
}

/// The words of a little-endian byte buffer (one copy into aligned memory);
/// `None` if its length is not a multiple of 8.
pub fn words_from_le_bytes(bytes: &[u8]) -> Option<Vec<i64>> {
    if bytes.len() % 8 != 0 {
        return None;
    }
    Some(bytemuck::pod_collect_to_vec(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn outputs_round_trip_through_words() {
        let value = (Q32::from_bits(-5), vec![1i64, -2, 3], Shaped { shape: vec![2, 1], values: vec![true, false] });
        let words = value.encode_to_vec();
        assert_eq!(words, vec![-5, 3, 1, -2, 3, 2, 2, 1, 2, 1, 0]);
        assert_eq!(decode_all::<(Q32, Vec<i64>, Shaped<bool>)>(&words), Ok(value));
    }

    #[test]
    fn malformed_words_are_rejected() {
        assert_eq!(decode_all::<Vec<i64>>(&[3, 1, 2]), Err(DecodeError));
        assert_eq!(decode_all::<i64>(&[1, 2]), Err(DecodeError));
        assert_eq!(decode_all::<bool>(&[2]), Err(DecodeError));
        assert_eq!(decode_all::<Vec<i64>>(&[-1]), Err(DecodeError));
    }

    #[test]
    fn bytes_round_trip() {
        let words = vec![0, -1, i64::MAX, i64::MIN, 42];
        assert_eq!(words_from_le_bytes(words_as_le_bytes(&words)), Some(words.clone()));
        assert_eq!(words_as_le_bytes(&[1, -2]), [1, 0, 0, 0, 0, 0, 0, 0, 254, 255, 255, 255, 255, 255, 255, 255]);
        assert_eq!(words_from_le_bytes(&[1, 2, 3]), None);
    }
}
