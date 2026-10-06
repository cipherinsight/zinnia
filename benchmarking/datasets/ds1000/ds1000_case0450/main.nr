// DS-1000 case 0450, Noir baseline.
// Kernel: `a[:, np.array(second).reshape(-1, 1), third]` on Float[7, 8, 9] with
// second, third Integer[2]: result[i, j, k] = a[i, second[j], third[k]], shape
// (7, 2, 2), NumPy index rules (negative indices wrap, others out of range
// raise IndexError: no proof).
// Each index is resolved once (`axis_pos`, a hint pinned by 4-bit range
// checks) and every output is a native array read. noir_hw::index_pos would
// use a 32-bit range check, whose range table costs more than this circuit.
// @outputs: float[28]

global D1: u32 = 8;
global D2: u32 = 9;

// (position, negative) of a NumPy index into an axis of length len.
unconstrained fn axis_pos_hint(idx: Field, len: Field) -> (Field, bool) {
    let neg = (idx as u128) >= 18446744073709551616; // idx < 0
    (if neg { idx + len } else { idx }, neg)
}

// The position P in [0, L) a NumPy index selects on an axis of length L <= 16.
fn axis_pos<let L: u32>(idx: Field) -> Field {
    // Safety: pinned by `check_axis_pos`.
    let (p, neg) = unsafe { axis_pos_hint(idx, L as Field) };
    check_axis_pos::<L>(idx, p, neg);
    p
}

// idx = P - L neg with P in [0, L) (P and L - 1 - P in [0, 16)). neg = false
// gives idx in [0, L), neg = true gives idx in [-L, -1]: the ranges are
// disjoint, so a valid idx has one witness and any other none.
fn check_axis_pos<let L: u32>(idx: Field, p: Field, neg: bool) {
    p.assert_max_bit_size::<4>();
    ((L as Field) - 1 - p).assert_max_bit_size::<4>();
    assert(idx == p - if neg { L as Field } else { 0 }, "index out of bounds");
}

fn main(a: [Field; 7 * D1 * D2], second: [Field; 2], third: [Field; 2]) -> pub [Field; 28] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(second);
    noir_hw::check_inputs(third);
    let p = [axis_pos::<D1>(second[0]), axis_pos::<D1>(second[1])];
    let q = [axis_pos::<D2>(third[0]), axis_pos::<D2>(third[1])];
    let mut out: [Field; 28] = [0; 28];
    for j in 0..2 {
        for k in 0..2 {
            // offset of (second[j], third[k]) inside a[i], below 72
            let pq = p[j] * (D2 as Field) + q[k];
            pq.assert_max_bit_size::<7>();
            let pq = pq as u32;
            for i in 0..7 {
                out[i * 4 + j * 2 + k] = a[i * D1 * D2 + pq];
            }
        }
    }
    out
}
