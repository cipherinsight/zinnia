// DS-1000 case 0375 -- Noir baseline.
// Sort grades, compute ys, find t, then return grades unchanged. Mirrors
// the kernel's dead computation faithfully.

use noir_fpc::{qlt, qdiv, q_from_int};

global N: u32 = 50;

fn main(grades: [Field; 50], threshold: Field) -> pub [Field; 50] {
    let mut xs: [Field; 50] = grades;
    for i in 0..N {
        for j in (i + 1)..N {
            if qlt(xs[j], xs[i]) {
                let t = xs[i]; xs[i] = xs[j]; xs[j] = t;
            }
        }
    }
    let n_q = q_from_int(N as u64);
    let mut ys: [Field; 50] = [0; 50];
    for k in 0..N {
        ys[k] = qdiv(q_from_int((k + 1) as u64), n_q);
    }
    let mut found: bool = false;
    let mut t_idx: u32 = 0;
    for k in 0..N {
        let hit = qlt(threshold, ys[k]);
        if hit & !found {
            t_idx = k;
            found = true;
        }
    }
    // low and high are discarded; touch xs to keep loop alive.
    let _low = xs[0];
    let mut _high = xs[0];
    for k in 1..N {
        if t_idx == k { _high = xs[k]; }
    }
    grades
}
