//! The public output of a guest: how a program's output is committed as
//! 64-bit words, decoded again by the hosts, and flattened into the values
//! the output check compares with NumPy.
//!
//! Layout: a scalar is its words ([`Element`]); a `Vec` is its length, then
//! its items; an ndarray is its dimensions, then its elements in row-major
//! order; a tuple is its items in order. The guest commits the words with
//! one raw call (SP1 `io::commit_slice`, RISC Zero `env::commit_slice`).

use alloc::vec::Vec;
use core::slice::Iter;

use ndarray::{Array, Dimension};
use q32::{C32, Q32};

use crate::io::Element;

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

/// A value a guest makes public (see the module documentation).
pub trait Publics: Sized {
    fn encode(&self, out: &mut Vec<i64>);
    fn decode(words: &mut Iter<'_, i64>) -> Result<Self, DecodeError>;
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

fn next_element<T: Element>(words: &mut Iter<'_, i64>) -> Result<T, DecodeError> {
    let slice = words.as_slice();
    if slice.len() < T::WORDS {
        return Err(DecodeError);
    }
    let value = T::from_words(&slice[..T::WORDS]).ok_or(DecodeError)?;
    words.nth(T::WORDS - 1);
    Ok(value)
}

fn next_length(words: &mut Iter<'_, i64>) -> Result<usize, DecodeError> {
    let len: i64 = next_element(words)?;
    usize::try_from(len).map_err(|_| DecodeError)
}

macro_rules! scalar_publics {
    ($($ty:ty => |$v:ident, $out:ident| $flatten:expr),* $(,)?) => {
        $(
            impl Publics for $ty {
                fn encode(&self, out: &mut Vec<i64>) {
                    self.push_words(out);
                }
                fn decode(words: &mut Iter<'_, i64>) -> Result<Self, DecodeError> {
                    next_element(words)
                }
                fn flatten_into(&self, $out: &mut Vec<Public>) {
                    let $v = *self;
                    $flatten
                }
            }
        )*
    };
}
scalar_publics! {
    Q32 => |v, out| out.push(Public::Float(v)),
    i64 => |v, out| out.push(Public::Int(v)),
    bool => |v, out| out.push(Public::Bool(v)),
    // NumPy's complex values are compared as (real, imaginary) pairs.
    C32 => |v, out| out.extend([Public::Float(v.re), Public::Float(v.im)]),
}

impl<T: Publics> Publics for Vec<T> {
    fn encode(&self, out: &mut Vec<i64>) {
        out.push(self.len() as i64);
        for item in self {
            item.encode(out);
        }
    }
    fn decode(words: &mut Iter<'_, i64>) -> Result<Self, DecodeError> {
        let len = next_length(words)?;
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

/// An ndarray of fixed rank: its dimensions, then its elements in row-major
/// (logical) order.
impl<T: Publics + Clone, D: Dimension> Publics for Array<T, D> {
    fn encode(&self, out: &mut Vec<i64>) {
        out.extend(self.shape().iter().map(|&d| d as i64));
        for item in self.iter() {
            item.encode(out);
        }
    }
    fn decode(words: &mut Iter<'_, i64>) -> Result<Self, DecodeError> {
        let rank = D::NDIM.ok_or(DecodeError)?;
        let mut shape = D::zeros(rank);
        for axis in 0..rank {
            shape[axis] = next_length(words)?;
        }
        let len = shape.size_checked().ok_or(DecodeError)?;
        if len > words.len() {
            return Err(DecodeError);
        }
        let items = (0..len).map(|_| T::decode(words)).collect::<Result<Vec<T>, _>>()?;
        Array::from_shape_vec(shape, items).map_err(|_| DecodeError)
    }
    fn flatten_into(&self, out: &mut Vec<Public>) {
        for item in self.iter() {
            item.flatten_into(out);
        }
    }
}

/// A value the kernel may not return (Python `None`): a 0 word, or a 1 word
/// and the value. `None` contributes no public values.
impl<T: Publics> Publics for Option<T> {
    fn encode(&self, out: &mut Vec<i64>) {
        match self {
            None => out.push(0),
            Some(value) => {
                out.push(1);
                value.encode(out);
            }
        }
    }
    fn decode(words: &mut Iter<'_, i64>) -> Result<Self, DecodeError> {
        if next_element::<bool>(words)? {
            Ok(Some(T::decode(words)?))
        } else {
            Ok(None)
        }
    }
    fn flatten_into(&self, out: &mut Vec<Public>) {
        if let Some(value) = self {
            value.flatten_into(out);
        }
    }
}

impl<T: Publics> Publics for Shaped<T> {
    fn encode(&self, out: &mut Vec<i64>) {
        self.shape.encode(out);
        self.values.encode(out);
    }
    fn decode(words: &mut Iter<'_, i64>) -> Result<Self, DecodeError> {
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
            fn decode(words: &mut Iter<'_, i64>) -> Result<Self, DecodeError> {
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
tuple_publics!(A, B, C, D, E);
tuple_publics!(A, B, C, D, E, F);

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
    use ndarray::{arr2, Array2};

    #[test]
    fn outputs_round_trip_through_words() {
        let value = (Q32::from_bits(-5), vec![1i64, -2, 3], Shaped { shape: vec![2, 1], values: vec![true, false] });
        let words = value.encode_to_vec();
        assert_eq!(words, vec![-5, 3, 1, -2, 3, 2, 2, 1, 2, 1, 0]);
        assert_eq!(decode_all::<(Q32, Vec<i64>, Shaped<bool>)>(&words), Ok(value));
    }

    #[test]
    fn arrays_and_complex_round_trip() {
        let a: Array2<i64> = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let z = C32::new(Q32::from_bits(7), Q32::from_bits(-8));
        let words = (a.clone(), z).encode_to_vec();
        assert_eq!(words, vec![2, 3, 1, 2, 3, 4, 5, 6, 7, -8]);
        assert_eq!(decode_all::<(Array2<i64>, C32)>(&words), Ok((a, z)));
        let flat = decode_all::<(Array2<i64>, C32)>(&words).unwrap().flatten();
        assert_eq!(flat.len(), 8);
        assert_eq!(flat[6], Public::Float(Q32::from_bits(7)));
    }

    #[test]
    fn malformed_words_are_rejected() {
        assert_eq!(decode_all::<Vec<i64>>(&[3, 1, 2]), Err(DecodeError));
        assert_eq!(decode_all::<i64>(&[1, 2]), Err(DecodeError));
        assert_eq!(decode_all::<bool>(&[2]), Err(DecodeError));
        assert_eq!(decode_all::<Vec<i64>>(&[-1]), Err(DecodeError));
        assert_eq!(decode_all::<Array2<i64>>(&[2, 2, 1, 2, 3]), Err(DecodeError));
        assert_eq!(decode_all::<C32>(&[1]), Err(DecodeError));
    }

    #[test]
    fn bytes_round_trip() {
        let words = vec![0, -1, i64::MAX, i64::MIN, 42];
        assert_eq!(words_from_le_bytes(words_as_le_bytes(&words)), Some(words.clone()));
        assert_eq!(words_as_le_bytes(&[1, -2]), [1, 0, 0, 0, 0, 0, 0, 0, 254, 255, 255, 255, 255, 255, 255, 255]);
        assert_eq!(words_from_le_bytes(&[1, 2, 3]), None);
    }
}
