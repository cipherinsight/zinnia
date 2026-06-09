// DS-1000 case 0462 -- Noir baseline.
// selection[i] = (i == 0 or a[i] != a[i-1]) and (a[i] != 0); result = a[selection].
// Output 21 fields: length + 20 padded buffer.

global N: u32 = 20;

fn main(a: [Field; 20], claimed_len: Field, claimed: [Field; 20]) -> pub [Field; 21] {
    let _ = claimed[0];
    let _ = claimed_len;
    let mut keep: [Field; 20] = [0; 20];
    for i in 0..N {
        let prev_eq: Field = if i == 0 { 0 } else { if a[i] == a[i - 1] { 1 } else { 0 } };
        let non_zero: Field = if a[i] == 0 { 0 } else { 1 };
        keep[i] = (1 - prev_eq) * non_zero;
    }
    let mut rank_after: [Field; 20] = [0; 20];
    let mut running: Field = 0;
    for i in 0..N {
        running = running + keep[i];
        rank_after[i] = running;
    }
    let total = rank_after[N - 1];
    let mut out_padded: [Field; 20] = [0; 20];
    for k in 0..N {
        let target = (k + 1) as Field;
        let mut chosen: Field = 0;
        for i in 0..N {
            let rank_eq: Field = if rank_after[i] == target { 1 } else { 0 };
            let pick = keep[i] * rank_eq;
            chosen = chosen + a[i] * pick;
        }
        out_padded[k] = chosen;
    }
    let mut out: [Field; 21] = [0; 21];
    out[0] = total;
    for k in 0..N { out[k + 1] = out_padded[k]; }
    out
}
