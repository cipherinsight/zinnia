// DS-1000 case 298 — Risc0 guest program.
// One-hot encode a Float[5] against its sorted-unique values.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [f64; 5] = [0.0; 5];
    for i in 0..5 {
        a[i] = env::read();
    }
    let claim_n_unique: u64 = env::read();
    let mut claim_b: Vec<Vec<f64>> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut row: Vec<f64> = Vec::with_capacity(claim_n_unique as usize);
        for _ in 0..claim_n_unique {
            row.push(env::read());
        }
        claim_b.push(row);
    }

    let mut vals: Vec<f64> = Vec::with_capacity(5);
    for i in 0..5 {
        let v = a[i];
        let mut present = false;
        for k in 0..vals.len() {
            if (vals[k] - v).abs() < 1e-12 {
                present = true;
                break;
            }
        }
        if !present {
            vals.push(v);
        }
    }
    let n_unique = vals.len();
    for i in 0..n_unique {
        for j in (i + 1)..n_unique {
            if vals[j] < vals[i] {
                let t = vals[i];
                vals[i] = vals[j];
                vals[j] = t;
            }
        }
    }
    let mut idx: [usize; 5] = [0; 5];
    for i in 0..5 {
        let mut k = 0usize;
        while k < n_unique && (vals[k] - a[i]).abs() >= 1e-12 {
            k += 1;
        }
        idx[i] = k;
    }
    let mut b: Vec<Vec<f64>> = Vec::with_capacity(5);
    for _ in 0..5 {
        b.push(vec![0.0; n_unique]);
    }
    for i in 0..5 {
        b[i][idx[i]] = 1.0;
    }

    assert_eq!(n_unique as u64, claim_n_unique);
    for i in 0..5 {
        for j in 0..n_unique {
            assert!((b[i][j] - claim_b[i][j]).abs() < 1e-9);
        }
    }
}
