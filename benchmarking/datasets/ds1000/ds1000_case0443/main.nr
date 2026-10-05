// DS-1000 case 0443 -- Noir baseline. A[np.in1d(A, B)] on Integer[20]x[3].
// Output 21: length + 20 padded buffer.

global NA: u32 = 20;
global NB: u32 = 3;

fn main(a: [Field; 20], b: [Field; 3],
        claimed_len: Field, claimed: [Field; 20]) -> pub [Field; 21] {
    let _ = claimed[0];
    let _ = claimed_len;
    let mut keep: [Field; 20] = [0; 20];
    for i in 0..NA {
        let mut any_eq: Field = 0;
        for j in 0..NB {
            let eq: Field = if a[i] == b[j] { 1 } else { 0 };
            any_eq = any_eq + eq - any_eq * eq;
        }
        keep[i] = any_eq;
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
