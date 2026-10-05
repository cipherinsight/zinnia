//! DS-1000 case 0425: the low `m = 11` bits of every entry of `a`, most
//! significant first.
//!
//!     def kernel(a: Integer[Array, "20"]):
//!         m = 11
//!         result = (((a[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
//!         return result
// output: [i64]

use crate::np::twos_complement;

const N: u32 = 20;
/// `1 << (m - 1)`, the mask of the first output column.
const TOP_BIT: u64 = 0x400;

/// Returns shape `(20, m)`, flattened row-major.
#[executable]
fn main(a: Array<i64>) -> Array<i64> {
    assert!(a.len() == N, "a has the wrong length");
    let mut bits = array![];
    for v in a {
        let word = twos_complement(v);
        let mut mask = TOP_BIT;
        while mask != 0 {
            bits.append(if word & mask > 0 {
                1
            } else {
                0
            });
            mask /= 2;
        }
    }
    bits
}
