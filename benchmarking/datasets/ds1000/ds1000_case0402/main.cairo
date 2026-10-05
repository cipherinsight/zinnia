//! DS-1000 case 0402: the Fortran-order linear index of `index` in an array
//! of shape `dims`.
//!
//!     def kernel(a: Float[Array, "8 9 8 8 8"], index: Integer[Array, "5"]):
//!         dims = np.array([8, 9, 8, 8, 8])
//!         result = np.ravel_multi_index(index, dims=dims, order='F')
//!         return result
// unused parameters: a
// output: i64

const DIMS: [i64; 5] = [8, 9, 8, 8, 8];

#[executable]
fn main(index: Array<i64>) -> i64 {
    assert!(index.len() == 5, "index has the wrong length");
    let dims = DIMS.span();
    // Fortran order: the first index varies fastest, so Horner's rule runs
    // from the last axis to the first.
    let mut flat = 0;
    let mut axis = 5;
    while axis > 0 {
        axis -= 1;
        let i = *index.at(axis);
        let dim = *dims.at(axis);
        assert!(0 <= i && i < dim, "invalid entry in coordinates array");
        flat = flat * dim + i;
    }
    flat
}
