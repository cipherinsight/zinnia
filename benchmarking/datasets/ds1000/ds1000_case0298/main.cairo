// DS-1000 case 298 — one-hot encoding via np.unique(return_inverse=True).
// kernel.py:
//     vals, idx = np.unique(a, return_inverse=True)
//     b = np.zeros((a.size, vals.size))
//     b[np.arange(a.size), idx] = 1
// Sample input: a = [1.5, -0.4, 1.3, 1.5, 1.3];
//   vals = [-0.4, 1.3, 1.5], idx = [2, 0, 1, 2, 1].
// Quantisation ×2^32 (Q) to match Zinnia's `precision_bits = 32`. Signed → i128.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let mut a: Array<i128> = ArrayTrait::new();
    a.append((15 * Q) / 10);          //  1.5
    a.append(-(4 * Q) / 10);           // -0.4
    a.append((13 * Q) / 10);          //  1.3
    a.append((15 * Q) / 10);          //  1.5
    a.append((13 * Q) / 10);          //  1.3

    // Build vals: unique values in order of first occurrence, then sort ascending.
    let mut vals: Array<i128> = ArrayTrait::new();
    for i in 0_usize..5 {
        let v: i128 = *a.at(i);
        let mut present: bool = false;
        let mut k: usize = 0;
        while k < vals.len() {
            if *vals.at(k) == v {
                present = true;
            }
            k += 1;
        };
        if !present {
            vals.append(v);
        }
    };
    let n_unique: usize = vals.len();
    // Insertion-sort into a fresh array (Array<i128> isn't mutable-in-place).
    let mut sorted: Array<i128> = ArrayTrait::new();
    let mut taken: Array<bool> = ArrayTrait::new();
    let mut t: usize = 0;
    while t < n_unique {
        taken.append(false);
        t += 1;
    };
    let mut written: usize = 0;
    while written < n_unique {
        let mut min_idx: usize = n_unique;
        let mut min_val: i128 = 0;
        let mut s: usize = 0;
        while s < n_unique {
            if !*taken.at(s) {
                let v = *vals.at(s);
                if min_idx == n_unique || v < min_val {
                    min_idx = s;
                    min_val = v;
                }
            }
            s += 1;
        };
        sorted.append(min_val);
        // Rebuild `taken` with `taken[min_idx] = true`.
        let mut new_taken: Array<bool> = ArrayTrait::new();
        let mut q: usize = 0;
        while q < n_unique {
            if q == min_idx {
                new_taken.append(true);
            } else {
                new_taken.append(*taken.at(q));
            }
            q += 1;
        };
        taken = new_taken;
        written += 1;
    };

    // idx[i] = position of a[i] in sorted.
    let mut idx: Array<usize> = ArrayTrait::new();
    for i in 0_usize..5 {
        let v: i128 = *a.at(i);
        let mut k: usize = 0;
        while k < n_unique && *sorted.at(k) != v {
            k += 1;
        };
        idx.append(k);
    };

    // Build b (5 × n_unique). b[i][idx[i]] = 1.0 → 1·Q.
    let mut b: Array<i128> = ArrayTrait::new();
    for i in 0_usize..5 {
        for j in 0_usize..n_unique {
            if j == *idx.at(i) {
                b.append(Q);
            } else {
                b.append(0);
            };
        };
    };

    // Claimed output (Q-scaled).
    let claim_n_unique: usize = 3;
    let claim_b: Array<i128> = array![
        0, 0, Q,
        Q, 0, 0,
        0, Q, 0,
        0, 0, Q,
        0, Q, 0,
    ];
    assert!(n_unique == claim_n_unique);
    let mut k: usize = 0;
    while k < 5 * n_unique {
        assert!(*b.at(k) == *claim_b.at(k));
        k += 1;
    };
}
