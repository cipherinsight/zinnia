//! DS-1000 case 0318: reorder the columns of `a` by the inverse of
//! `permutation`.
//!
//!     def kernel(a: Float[Array, "8 6"], permutation: Integer[Array, "6"]):
//!         c = np.empty_like(permutation)
//!         c[permutation] = np.arange(len(permutation))
//!         a = a[:, c]
//!         return a
// output: [q32]
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::np::np_index;
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>, permutation: Array<i64>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS && permutation.len() == COLS, "inputs have the wrong length");
    let c = inverse_permutation(permutation.span()).span();
    let mut result = array![];
    for r in 0..ROWS {
        for j in c {
            result.append(*a.at(r * COLS + *j));
        }
    }
    result
}

/// `c[permutation] = np.arange(n)`. An entry that no index writes would keep
/// `np.empty_like`'s uninitialized value, so `permutation` must hit every
/// position, i.e. be a permutation (negative entries count from the end).
fn inverse_permutation(permutation: Span<i64>) -> Array<u32> {
    let n = permutation.len();
    // `i + 1` at key `permutation[i]`; the default 0 marks an unwritten entry.
    let mut written: Felt252Dict<u32> = Default::default();
    for i in 0..n {
        written.insert(np_index(*permutation.at(i), n).into(), i + 1);
    }
    let mut c = array![];
    for j in 0..n {
        let entry = written.get(j.into());
        assert!(entry != 0, "permutation leaves an entry of c uninitialized");
        c.append(entry - 1);
    }
    c
}
