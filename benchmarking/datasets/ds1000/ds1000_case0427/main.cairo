//! DS-1000 case 0427: the low `m = 11` bits of the XOR of all entries of `a`,
//! most significant first.
//!
//!     def kernel(a: Integer[Array, "20"]):
//!         m = 11
//!         res = np.array([0])
//!         for i in a:
//!             res = res ^ i
//!         result = (((res[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
//!         return result
// output: [i64]

use crate::np::twos_complement;

const N: u32 = 20;
/// `1 << (m - 1)`, the mask of the first output column.
const TOP_BIT: u64 = 0x400;

/// Returns shape `(1, m)`, flattened row-major.
#[executable]
fn main(a: Array<i64>) -> Array<i64> {
    assert!(a.len() == N, "a has the wrong length");
    let mut res: u64 = 0;
    for v in a {
        res = res ^ twos_complement(v);
    }
    let mut bits = array![];
    let mut mask = TOP_BIT;
    while mask != 0 {
        bits.append(if res & mask > 0 {
            1
        } else {
            0
        });
        mask /= 2;
    }
    bits
}
