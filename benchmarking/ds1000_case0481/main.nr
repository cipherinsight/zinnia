// DS-1000 case 0481 -- Noir baseline.
// indices where (x==a) & (y==b). Output: length + 9 padded buffer.

global N: u32 = 9;

fn main(x: [Field; 9], y: [Field; 9], a: Field, b_val: Field,
        claimed_len: Field, claimed: [Field; 9]) -> pub [Field; 10] {
    let _ = claimed[0];
    let _ = claimed_len;
    let mut keep: [Field; 9] = [0; 9];
    for k in 0..N {
        let m = (x[k] == a) & (y[k] == b_val);
        keep[k] = if m { 1 } else { 0 };
    }
    let mut rank_after: [Field; 9] = [0; 9];
    let mut running: Field = 0;
    for i in 0..N {
        running = running + keep[i];
        rank_after[i] = running;
    }
    let total = rank_after[N - 1];
    let mut out_padded: [Field; 9] = [0; 9];
    for k in 0..N {
        let target = (k + 1) as Field;
        let mut chosen: Field = 0;
        for i in 0..N {
            let rank_eq: Field = if rank_after[i] == target { 1 } else { 0 };
            let pick = keep[i] * rank_eq;
            chosen = chosen + (i as Field) * pick;
        }
        out_padded[k] = chosen;
    }
    let mut out: [Field; 10] = [0; 10];
    out[0] = total;
    for k in 0..N { out[k + 1] = out_padded[k]; }
    out
}
