// DS-1000 case 0298: literal translation of kernel.py (LITERAL_BASELINES.md).
// Unique uses stable bubble sort, adjacent equality and equality-test compaction.
// Dynamic-width matrix is padded internally and packed for public output.
// @outputs: dynamic2 float[5x5]

use noir_hw::{window, SCALE};

global N: u32 = 5;

fn main(a: [Field; N]) -> pub [Field; 2 + N * N] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);

    // vals, idx = np.unique(a, return_inverse=True)
    // Sort a ascending, carrying the original index (bubble sort, stable,
    // N * (N - 1) / 2 compare-and-swaps total).
    let mut v: [Field; N] = a;
    let mut p: [Field; N] = [0; N];
    for i in 0..N {
        p[i] = i as Field;
    }
    for pass in 0..N - 1 {
        for k in 0..N - 1 - pass {
            let gt = window(v[k]) > window(v[k + 1]);
            if gt {
                let tv = v[k];
                v[k] = v[k + 1];
                v[k + 1] = tv;
                let tp = p[k];
                p[k] = p[k + 1];
                p[k + 1] = tp;
            }
        }
    }

    // first[j] = 1 when sorted position j starts a new distinct value.
    let mut first: [Field; N] = [0; N];
    first[0] = 1;
    for j in 1..N {
        first[j] = (v[j] != v[j - 1]) as Field;
    }
    // rank_sorted[j] = (running count of distinct values through j) - 1.
    let mut rank_sorted: [Field; N] = [0; N];
    let mut running: Field = 0;
    for j in 0..N {
        running += first[j];
        rank_sorted[j] = running - 1;
    }
    let n_unique = running; // vals.size
    let mut vals = [0; N];
    for slot in 0..N {
        for j in 0..N {
            vals[slot] += ((first[j] == 1) & (rank_sorted[j] == slot as Field)) as Field * v[j];
        }
    }


    // idx[i] (original order) = rank_sorted[j] for the sorted position j
    // carrying original index i.
    let mut idx: [Field; N] = [0; N];
    for i in 0..N {
        let mut acc: Field = 0;
        for j in 0..N {
            acc += ((p[j] == i as Field) as Field) * rank_sorted[j];
        }
        idx[i] = acc;
    }

    // b = zeros((a.size, vals.size)); scatter at each private column.
    let mut b = [0; N * N];
    for i in 0..N {
        for c in 0..N {
            b[i * N + c] = if idx[i] == c as Field { SCALE } else { b[i * N + c] };
        }
    }
    // Pack the padded matrix at its dynamic row stride for dynamic2 output.
    let mut out = [0; 2 + N * N];
    out[0] = N as Field;
    out[1] = n_unique;
    for t in 0..N * N {
        for m in 1..N + 1 {
            if t < N * m {
                out[2 + t] += (n_unique == m as Field) as Field * b[(t / m) * N + t % m];
            }
        }
    }
    out
}
