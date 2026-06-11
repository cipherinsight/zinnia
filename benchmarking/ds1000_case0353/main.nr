// DS-1000 case 0353 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// Setdiff of A's rows from B's rows via ravel_multi_index + in1d.

use noir_fpc::qgt;

fn main(a: [Field; 6], b: [Field; 21]) -> pub [Field; 8] {
    let mut dims: [Field; 3] = [0; 3];
    for k in 0..3 {
        let mut m = a[k];
        for i in 1..2 {
            if qgt(a[i * 3 + k], m) { m = a[i * 3 + k]; }
        }
        for i in 0..7 {
            if qgt(b[i * 3 + k], m) { m = b[i * 3 + k]; }
        }
        dims[k] = m + 1;
    }
    // Soundness: ravel_multi_index requires each component < dim[axis].
    // Skipping these is a soundness hole - an adversarial witness with
    // a[i,k] >= dims[k] can alias into a different b ravel position,
    // marking a row as "in B" incorrectly.
    for i in 0..2 {
        for k in 0..3 {
            assert((a[i * 3 + k] as u128) < (dims[k] as u128));
        }
    }
    for i in 0..7 {
        for k in 0..3 {
            assert((b[i * 3 + k] as u128) < (dims[k] as u128));
        }
    }
    let mut a_idx: [Field; 2] = [0; 2];
    for r in 0..2 {
        let t0 = a[r * 3] * dims[1];
        let t1 = t0 + a[r * 3 + 1];
        let t2 = t1 * dims[2];
        a_idx[r] = t2 + a[r * 3 + 2];
    }
    let mut b_idx: [Field; 7] = [0; 7];
    for r in 0..7 {
        let t0 = b[r * 3] * dims[1];
        let t1 = t0 + b[r * 3 + 1];
        let t2 = t1 * dims[2];
        b_idx[r] = t2 + b[r * 3 + 2];
    }
    // Compact output (mirrors Zinnia's dyn-array encoding): kept rows
    // packed to the front in order, zero-padded; metadata = runtime
    // length (rows * 3) + row count.
    let mut keeps: [Field; 2] = [0; 2];
    for r in 0..2 {
        let mut in_b: Field = 0;
        for s in 0..7 {
            let eq: Field = if a_idx[r] == b_idx[s] { 1 } else { 0 };
            in_b = in_b + eq - in_b * eq;
        }
        keeps[r] = 1 - in_b;
    }
    let count = keeps[0] + keeps[1];
    let both = keeps[0] * keeps[1];
    let mut out: [Field; 8] = [0; 8];
    out[0] = count * 3;
    out[1] = count;
    for j in 0..3 {
        // Slot row 0: A0 if kept, else A1 if kept, else 0.
        let fallback = keeps[1] * a[3 + j];
        out[2 + j] = keeps[0] * a[j] + (1 - keeps[0]) * fallback;
        // Slot row 1: A1 only when both rows kept.
        out[5 + j] = both * a[3 + j];
    }
    out
}
