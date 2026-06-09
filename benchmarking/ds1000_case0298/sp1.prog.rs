// DS-1000 case 298 — SP1 guest program.
// Kernel: one-hot encode `a` against its sorted-unique values.
//   vals, idx = np.unique(a, return_inverse=True)
//   b = np.zeros((a.size, vals.size))
//   b[np.arange(a.size), idx] = 1
// Inputs: Float[5] + claimed (n_unique, b matrix). Sample uses test_case_id=3:
//   a = [1.5, -0.4, 1.3, 1.5, 1.3] → vals = [-0.4, 1.3, 1.5], idx = [2, 0, 1, 2, 1].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [f64; 5] = [0.0; 5];
    for i in 0..5 {
        a[i] = sp1_zkvm::io::read::<f64>();
    }
    let claim_n_unique: u64 = sp1_zkvm::io::read::<u64>();
    let mut claim_b: Vec<Vec<f64>> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut row: Vec<f64> = Vec::with_capacity(claim_n_unique as usize);
        for _ in 0..claim_n_unique {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        claim_b.push(row);
    }

    // Compute sorted unique values via insertion-sort style. n_unique <= 5.
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
    // Sort ascending.
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
    // idx[i] = index of a[i] in vals.
    let mut idx: [usize; 5] = [0; 5];
    for i in 0..5 {
        let mut k = 0usize;
        while k < n_unique && (vals[k] - a[i]).abs() >= 1e-12 {
            k += 1;
        }
        idx[i] = k;
    }
    // Build b.
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
