//! NPBench spmv: sparse matrix (CSR: row pointers, column indices, values)
//! times dense vector.
//!
//! ```python
//! def spmv(A_row: Integer[Array, '4097'], A_col: Integer[Array, '8192'], A_val: Float[Array, '8192'], x: Float[Array, '4096']):
//!     y = np.empty(A_row.size - 1, A_val.dtype)
//!     for i in range(A_row.size - 1):
//!         cols = A_col[A_row[i]:A_row[i + 1]]
//!         vals = A_val[A_row[i]:A_row[i + 1]]
//!         y[i] = vals @ x[cols]
//!     return y
//! ```

use crate::prelude::*;

pub mod spmv {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let a_row: Array1<i64> = args.array1();
        let a_col: Array1<i64> = args.array1();
        let a_val: Array1<Q32> = args.array1();
        let x: Array1<Q32> = args.array1();
        assert!(!a_row.is_empty(), "negative dimensions are not allowed");
        (0..a_row.len() - 1)
            .map(|i| {
                let (start, stop) = (a_row[i], a_row[i + 1]);
                let cols = a_col.slice(s![np::slice_range(start, stop, a_col.len())]);
                let vals = a_val.slice(s![np::slice_range(start, stop, a_val.len())]);
                assert_eq!(vals.len(), cols.len(), "shapes not aligned");
                // `vals @ x[cols]`: one fused dot product of the gathered entries.
                let gather = |(&v, &c): (&Q32, &i64)| (v, x[np_index(c, x.len())]);
                dot(vals.iter().zip(&cols).map(gather))
            })
            .collect()
    }
}
