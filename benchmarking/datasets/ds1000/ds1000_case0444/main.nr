// DS-1000 case 0444 -- Noir baseline.
// A[(A>B[0] & A<B[1]) | (A>B[1] & A<B[2])] on Integer[20] x Integer[3].

use noir_fpc::qlt;

global NA: u32 = 20;

fn main(a: [Field; 20], b: [Field; 3],
        claimed_len: Field, claimed: [Field; 20]) -> pub [Field; 21] {
    let _ = claimed[0];
    let _ = claimed_len;
    let mut keep: [Field; 20] = [0; 20];
    for i in 0..NA {
        let in1 = qlt(b[0], a[i]) & qlt(a[i], b[1]);
        let in2 = qlt(b[1], a[i]) & qlt(a[i], b[2]);
        keep[i] = if in1 | in2 { 1 } else { 0 };
    }
    let mut rank_after: [Field; 20] = [0; 20];
    let mut running: Field = 0;
    for i in 0..NA {
        running = running + keep[i];
        rank_after[i] = running;
    }
    let total = rank_after[NA - 1];
    let mut out_padded: [Field; 20] = [0; 20];
    for k in 0..NA {
        let target = (k + 1) as Field;
        let mut chosen: Field = 0;
        for i in 0..NA {
            let rank_eq: Field = if rank_after[i] == target { 1 } else { 0 };
            let pick = keep[i] * rank_eq;
            chosen = chosen + a[i] * pick;
        }
        out_padded[k] = chosen;
    }
    let mut out: [Field; 21] = [0; 21];
    out[0] = total;
    for k in 0..NA { out[k + 1] = out_padded[k]; }
    out
}
