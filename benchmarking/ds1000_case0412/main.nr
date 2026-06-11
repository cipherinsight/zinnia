// DS-1000 case 0412 -- Noir baseline (compact-output version).
// x[x >= 0] on Float[10].
// Output 12 fields: count twice (runtime length + shape dim, mirroring
// Zinnia's dyn-array metadata) + 10 compacted cells (kept values packed
// to the front in source order, zero-padded). Compaction via a one-hot
// shift network.

use noir_fpc::is_negative;

global N: u32 = 10;

fn main(x: [Field; 10]) -> pub [Field; 12] {
    let mut keeps: [Field; 10] = [0; 10];
    let mut count: Field = 0;
    for i in 0..N {
        let neg = is_negative(x[i]);
        keeps[i] = if neg { 0 } else { 1 };
        count = count + keeps[i];
    }

    // One-hot shift compaction: `hot` marks the next free output slot;
    // deposit on keep, then shift right by one iff kept.
    let mut outs: [Field; 10] = [0; 10];
    let mut hot: [Field; 10] = [0; 10];
    hot[0] = 1;
    for i in 0..N {
        for j in 0..N {
            let here = hot[j] * keeps[i];
            outs[j] = here * x[i] + (1 - here) * outs[j];
        }
        let mut newhot: [Field; 10] = [0; 10];
        newhot[0] = hot[0] * (1 - keeps[i]);
        for j in 1..N {
            newhot[j] = keeps[i] * hot[j - 1] + (1 - keeps[i]) * hot[j];
        }
        hot = newhot;
    }

    let mut out: [Field; 12] = [0; 12];
    out[0] = count;
    out[1] = count;
    for j in 0..N {
        out[j + 2] = outs[j];
    }
    out
}
