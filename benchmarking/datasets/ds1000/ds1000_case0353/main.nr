// DS-1000 case 0353, Noir baseline.
// Kernel: `dims = np.maximum(B.max(0), A.max(0)) + 1;
// A[~np.in1d(np.ravel_multi_index(A.T, dims), np.ravel_multi_index(B.T, dims))]`
// on A: Integer[2, 3], B: Integer[7, 3]: the rows of A that are not rows of B
// (data-dependent number of rows).
// `ravel_multi_index` raises ValueError (no proof) for a negative coordinate
// and when prod(dims) exceeds 2^63 - 1; both are checked (each entry and the
// product in [0, 2^63)). The flat index of a row is then
// (x0 d1 + x1) d2 + x2, an exact integer, injective on the rows.
// Output: the number of kept rows, then the kept rows of A, zero-padded to
// 2 rows of 3.
// @outputs: dynamic int[2x3]

use noir_hw::max_of;

fn main(A: [Field; 6], B: [Field; 21]) -> pub [Field; 7] {
    for x in A {
        x.assert_max_bit_size::<63>();
    }
    for x in B {
        x.assert_max_bit_size::<63>();
    }
    let mut dims: [Field; 3] = [0; 3];
    for c in 0..3 {
        let mut column: [Field; 9] = [0; 9];
        for r in 0..2 {
            column[r] = A[r * 3 + c];
        }
        for r in 0..7 {
            column[2 + r] = B[r * 3 + c];
        }
        dims[c] = max_of(column) + 1;
    }
    (dims[0] * dims[1] * dims[2]).assert_max_bit_size::<63>();

    let mut flat_b: [Field; 7] = [0; 7];
    for r in 0..7 {
        flat_b[r] = (B[r * 3] * dims[1] + B[r * 3 + 1]) * dims[2] + B[r * 3 + 2];
    }
    let mut keep: [bool; 2] = [true; 2];
    for r in 0..2 {
        let flat_a = (A[r * 3] * dims[1] + A[r * 3 + 1]) * dims[2] + A[r * 3 + 2];
        for s in 0..7 {
            keep[r] &= flat_a != flat_b[s];
        }
    }

    // Kept rows to the front: slot 0 holds A[0] if kept, else A[1] if kept;
    // slot 1 holds A[1] if both are kept.
    let (k0, k1) = (keep[0] as Field, keep[1] as Field);
    let mut out: [Field; 7] = [0; 7];
    out[0] = k0 + k1;
    for c in 0..3 {
        out[1 + c] = k0 * A[c] + (1 - k0) * k1 * A[3 + c];
        out[4 + c] = k0 * k1 * A[3 + c];
    }
    out
}
