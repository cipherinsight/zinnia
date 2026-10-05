//! The private input of a guest.
//!
//! Every kernel parameter travels as one buffer of 64-bit words, through the
//! zkVMs' raw-slice I/O (one copy per buffer) rather than their serde codecs,
//! which cost hundreds of cycles per element (measured on RISC Zero: 287
//! cycles per element to read an array with `env::read`). A value is one or
//! more words ([`Element`]); an array is its elements in row-major order.
//!
//! Fixed-shape programs (the DS-1000 performance set) know their shapes, so
//! an array buffer holds only the elements. Shape-generic programs (NPBench,
//! Pythran) read every array through [`Args`]: the buffer starts with the
//! array's dimensions, and [`Args`] records them so the guest commits the
//! input shapes with its output. The proof then binds the shapes, as
//! Zinnia's circuit binds them by fixing them at compile time.

use alloc::vec::Vec;

use ndarray::{Array, Dimension, Ix1, Ix2, Ix3, Ix4};
use q32::{C32, Q32};

/// A value that travels as a fixed number of 64-bit words: a Q32 by its
/// encoding, an integer as itself, a boolean as 0 or 1, a complex number as
/// its real then its imaginary part.
pub trait Element: Copy {
    const WORDS: usize;
    /// `None` if `words` (exactly [`Element::WORDS`] of them) encode no
    /// value of this type.
    fn from_words(words: &[i64]) -> Option<Self>;
    fn push_words(self, out: &mut Vec<i64>);
}

impl Element for Q32 {
    const WORDS: usize = 1;
    fn from_words(words: &[i64]) -> Option<Q32> {
        Some(Q32::from_bits(words[0]))
    }
    fn push_words(self, out: &mut Vec<i64>) {
        out.push(self.to_bits());
    }
}

impl Element for i64 {
    const WORDS: usize = 1;
    fn from_words(words: &[i64]) -> Option<i64> {
        Some(words[0])
    }
    fn push_words(self, out: &mut Vec<i64>) {
        out.push(self);
    }
}

impl Element for bool {
    const WORDS: usize = 1;
    fn from_words(words: &[i64]) -> Option<bool> {
        match words[0] {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    fn push_words(self, out: &mut Vec<i64>) {
        out.push(self as i64);
    }
}

impl Element for C32 {
    const WORDS: usize = 2;
    fn from_words(words: &[i64]) -> Option<C32> {
        Some(C32::new(Q32::from_bits(words[0]), Q32::from_bits(words[1])))
    }
    fn push_words(self, out: &mut Vec<i64>) {
        out.push(self.re.to_bits());
        out.push(self.im.to_bits());
    }
}

/// Decodes a buffer of words into elements.
fn elements<T: Element>(words: &[i64]) -> Vec<T> {
    assert_eq!(words.len() % T::WORDS, 0, "buffer is not a whole number of elements");
    words
        .chunks_exact(T::WORDS)
        .map(|w| T::from_words(w).expect("buffer holds a value of the wrong type"))
        .collect()
}

/// The private input stream of a guest: the kernel parameters in signature
/// order, one buffer of words each.
pub trait Input {
    /// The next parameter's words.
    fn read_words(&mut self) -> Vec<i64>;

    /// A scalar parameter.
    fn read<T: Element>(&mut self) -> T {
        let words = self.read_words();
        assert_eq!(words.len(), T::WORDS, "a scalar parameter is one value");
        T::from_words(&words).expect("parameter has the wrong type")
    }

    /// A flattened array that must have exactly `len` elements.
    fn read_vec<T: Element>(&mut self, len: usize) -> Vec<T> {
        let values = elements(&self.read_words());
        assert_eq!(values.len(), len, "input array has the wrong length");
        values
    }

    /// A one-dimensional array of fixed length.
    fn read_array<T: Element, const N: usize>(&mut self) -> [T; N] {
        match self.read_vec(N).try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("read_vec checked the length"),
        }
    }

    /// A two-dimensional array of fixed shape `R x C`.
    fn read_matrix<T: Element, const R: usize, const C: usize>(&mut self) -> [[T; C]; R] {
        let flat = self.read_vec::<T>(R * C);
        core::array::from_fn(|i| core::array::from_fn(|j| flat[i * C + j]))
    }
}

/// The input of a shape-generic program. Array parameters carry their
/// dimensions; every dimension read is recorded so the guest can commit the
/// input shapes.
pub struct Args<'a, I: Input> {
    input: &'a mut I,
    shapes: Vec<i64>,
}

impl<'a, I: Input> Args<'a, I> {
    pub fn new(input: &'a mut I) -> Args<'a, I> {
        Args { input, shapes: Vec::new() }
    }

    /// A scalar parameter.
    pub fn scalar<T: Element>(&mut self) -> T {
        self.input.read()
    }

    /// An array parameter of rank `D::NDIM`: its dimensions, then its
    /// elements in row-major order.
    pub fn array<T: Element, D: Dimension>(&mut self) -> Array<T, D> {
        let rank = D::NDIM.expect("arrays have a fixed rank");
        let words = self.input.read_words();
        assert!(words.len() >= rank, "array parameter lacks its shape");
        let (dims, data) = words.split_at(rank);
        let mut shape = D::zeros(rank);
        for (axis, &extent) in dims.iter().enumerate() {
            shape[axis] = usize::try_from(extent).expect("negative array extent");
        }
        self.shapes.extend_from_slice(dims);
        Array::from_shape_vec(shape, elements(data)).expect("array data does not fill its shape")
    }

    pub fn array1<T: Element>(&mut self) -> Array<T, Ix1> {
        self.array()
    }

    pub fn array2<T: Element>(&mut self) -> Array<T, Ix2> {
        self.array()
    }

    pub fn array3<T: Element>(&mut self) -> Array<T, Ix3> {
        self.array()
    }

    pub fn array4<T: Element>(&mut self) -> Array<T, Ix4> {
        self.array()
    }

    /// The dimensions of every array read so far, in order.
    pub fn into_shapes(self) -> Vec<i64> {
        self.shapes
    }
}
