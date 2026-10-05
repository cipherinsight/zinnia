// DS-1000 case 0442 -- Noir baseline (compact-output version).
// A[~np.in1d(A, B)] on Integer[20]x[3].
// Output 22 fields: count twice (runtime length + shape dim, mirroring
// Zinnia's dyn-array metadata) + 20 compacted cells (kept values packed
// to the front in source order, zero-padded). Compaction via a one-hot
// shift network; entries beyond the reachable prefix are compile-time
// zeros, so the unrolled extra terms fold away.

global NA: u32 = 20;
global NB: u32 = 3;

fn main(a: [Field; 20], b: [Field; 3]) -> pub [Field; 22] {
    let mut keeps: [Field; 20] = [0; 20];
    let mut count: Field = 0;
    for i in 0..NA {
        let mut any_eq: Field = 0;
        for j in 0..NB {
            let eq: Field = if a[i] == b[j] { 1 } else { 0 };
            any_eq = any_eq + eq - any_eq * eq;
        }
        keeps[i] = 1 - any_eq;
        count = count + keeps[i];
    }

    // One-hot shift compaction: `hot` marks the next free output slot;
    // deposit on keep, then shift right by one iff kept.
    let mut outs: [Field; 20] = [0; 20];
    let mut hot: [Field; 20] = [0; 20];
    hot[0] = 1;
    for i in 0..NA {
        for j in 0..NA {
            let here = hot[j] * keeps[i];
            outs[j] = here * a[i] + (1 - here) * outs[j];
        }
        let mut newhot: [Field; 20] = [0; 20];
        newhot[0] = hot[0] * (1 - keeps[i]);
        for j in 1..NA {
            newhot[j] = keeps[i] * hot[j - 1] + (1 - keeps[i]) * hot[j];
        }
        hot = newhot;
    }

    let mut out: [Field; 22] = [0; 22];
    out[0] = count;
    out[1] = count;
    for j in 0..NA {
        out[j + 2] = outs[j];
    }
    out
}
