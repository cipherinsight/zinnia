//! Shape-generic guest I/O. One buffer per parameter (tuple items expanded),
//! array dimensions last, exactly as zkvm_common::Args. Public words start
//! with the length and contents of the input-shape vector.
use crate::q32::{Q32, C32};

pub trait Word<T> {
    fn read(ref words: Span<i64>) -> T;
    fn write(value: T, ref out: Array<i64>);
}
pub impl IntWord of Word<i64> {
    fn read(ref words: Span<i64>) -> i64 { *words.pop_front().unwrap() }
    fn write(value: i64, ref out: Array<i64>) { out.append(value); }
}
pub impl Q32Word of Word<Q32> {
    fn read(ref words: Span<i64>) -> Q32 { Q32 { bits: *words.pop_front().unwrap() } }
    fn write(value: Q32, ref out: Array<i64>) { out.append(value.bits); }
}
pub impl BoolWord of Word<bool> {
    fn read(ref words: Span<i64>) -> bool {
        let v = *words.pop_front().unwrap();
        assert!(v == 0 || v == 1, "invalid boolean");
        v == 1
    }
    fn write(value: bool, ref out: Array<i64>) { out.append(if value { 1 } else { 0 }); }
}
pub impl ComplexWord of Word<C32> {
    fn read(ref words: Span<i64>) -> C32 {
        let re = Word::read(ref words);
        let im = Word::read(ref words);
        C32 { re, im }
    }
    fn write(value: C32, ref out: Array<i64>) {
        out.append(value.re.bits); out.append(value.im.bits);
    }
}

#[derive(Drop)]
pub struct Args { buffers: Array<Array<i64>>, shapes: Array<i64> }

#[generate_trait]
pub impl ArgsImpl of ArgsTrait {
    fn new(buffers: Array<Array<i64>>) -> Args { Args { buffers, shapes: array![] } }

    fn scalar<T, +Word<T>, +Drop<T>>(ref self: Args) -> T {
        let buffer = self.buffers.pop_front().unwrap();
        let mut words = buffer.span();
        let value = Word::read(ref words);
        assert!(words.is_empty(), "scalar buffer length");
        value
    }

    fn array<T, +Word<T>, +Drop<T>>(ref self: Args, rank: u32) -> (Array<u32>, Array<T>) {
        let buffer = self.buffers.pop_front().unwrap();
        assert!(buffer.len() >= rank, "missing array dimensions");
        let end = buffer.len() - rank;
        let mut dims = array![];
        let mut size = 1_u32;
        for axis in end..buffer.len() {
            let dim = *buffer.at(axis);
            let extent: u32 = dim.try_into().unwrap();
            self.shapes.append(dim);
            dims.append(extent);
            size *= extent;
        };
        let mut words = buffer.span().slice(0, end);
        let mut values = array![];
        for _ in 0..size { values.append(Word::read(ref words)); };
        assert!(words.is_empty(), "array data does not fill shape");
        (dims, values)
    }

    fn array1<T, +Word<T>, +Drop<T>>(ref self: Args) -> Array<T> {
        let (_, values) = self.array(1); values
    }
    fn array2<T, +Word<T>, +Drop<T>>(ref self: Args) -> (u32, u32, Array<T>) {
        let (dims, values) = self.array(2); (*dims.at(0), *dims.at(1), values)
    }
    fn array3<T, +Word<T>, +Drop<T>>(ref self: Args) -> (u32, u32, u32, Array<T>) {
        let (dims, values) = self.array(3); (*dims.at(0), *dims.at(1), *dims.at(2), values)
    }
    fn array4<T, +Word<T>, +Drop<T>>(ref self: Args) -> (u32, u32, u32, u32, Array<T>) {
        let (dims, values) = self.array(4);
        (*dims.at(0), *dims.at(1), *dims.at(2), *dims.at(3), values)
    }

    /// Finish reading before appending outputs. Reject unread parameters.
    fn output(self: Args) -> Array<i64> {
        assert!(self.buffers.is_empty(), "unread parameters");
        let mut out = array![self.shapes.len().into()];
        for dim in self.shapes { out.append(dim); };
        out
    }
}

/// Encode an ndarray: extents, then elements, with no extra data length.
pub fn tensor<T, +Word<T>, +Copy<T>, +Drop<T>>(ref out: Array<i64>, dims: Span<u32>, values: Span<T>) {
    let mut size = 1;
    for dim in dims { size *= *dim; out.append((*dim).into()); };
    assert!(size == values.len(), "output shape mismatch");
    for value in values { Word::write(*value, ref out); };
}

/// Rust Vec output: length followed by elements.
pub fn vector<T, +Word<T>, +Copy<T>, +Drop<T>>(ref out: Array<i64>, values: Span<T>) {
    out.append(values.len().into());
    for value in values { Word::write(*value, ref out); };
}

#[cfg(test)]
mod tests {
    use super::{ArgsTrait, tensor, Word};
    use crate::q32::{Q32, C32};

    #[test]
    fn reads_parameters_and_commits_shapes_in_order() {
        let mut args = ArgsTrait::new(array![array![7], array![1, 2, 3, 4, 2, 2], array![1]]);
        let n: i64 = args.scalar();
        assert!(n == 7);
        let (r, c, a): (u32, u32, Array<i64>) = args.array2();
        assert!(r == 2 && c == 2 && a == array![1, 2, 3, 4]);
        let flag: bool = args.scalar();
        assert!(flag);
        let mut out = args.output();
        tensor(ref out, array![r, c].span(), a.span());
        assert!(out == array![2, 2, 2, 2, 2, 1, 2, 3, 4]);
    }

    #[test]
    fn empty_arrays_and_complex_words() {
        let mut args = ArgsTrait::new(array![array![0, 5], array![1, -2, 1]]);
        let (r, c, empty): (u32, u32, Array<Q32>) = args.array2();
        assert!(r == 0 && c == 5 && empty.is_empty());
        let complex: Array<C32> = args.array1();
        let z = *complex.at(0);
        assert!(z.re.bits == 1 && z.im.bits == -2);
        let mut out = args.output();
        Word::write(z, ref out);
        assert!(out == array![3, 0, 5, 1, 1, -2]);
    }

    #[test]
    #[should_panic]
    fn rejects_wrong_shape() {
        let mut args = ArgsTrait::new(array![array![10, 2]]);
        let _: Array<i64> = args.array1();
    }
    #[test]
    #[should_panic]
    fn rejects_negative_dimension() {
        let mut args = ArgsTrait::new(array![array![-1]]);
        let _: Array<i64> = args.array1();
    }
    #[test]
    #[should_panic]
    fn rejects_unread_parameters() {
        let args = ArgsTrait::new(array![array![1]]);
        let _ = args.output();
    }
    #[test]
    #[should_panic]
    fn rejects_non_boolean_words() {
        let mut args = ArgsTrait::new(array![array![2]]);
        let _: bool = args.scalar();
    }
}
