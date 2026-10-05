//! DS-1000 case 0403: the row-major (C-order) linear index of `index` in an
//! array of shape `dims`.
//!
//!     def kernel(a: Float[Array, "8 9 8 8 8"], index: Integer[Array, "5"]):
//!         dims = np.array([8, 9, 8, 8, 8])
//!         result = np.ravel_multi_index(index, dims=dims, order='C')
//!         return result
// unused parameters: a
// output: i64

const DIMS: [i64; 5] = [8, 9, 8, 8, 8];

#[executable]
fn main(index: Array<i64>) -> i64 {
    assert!(index.len() == 5, "index has the wrong length");
    let dims = DIMS.span();
    // C order: the last index varies fastest (Horner's rule from the first
    // axis).
    let mut flat = 0;
    for axis in 0..5_u32 {
        let i = *index.at(axis);
        let dim = *dims.at(axis);
        assert!(0 <= i && i < dim, "invalid entry in coordinates array");
        flat = flat * dim + i;
    }
    flat
}
