// DS-1000 case 0298, Noir baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Kernel: vals, idx = np.unique(a, return_inverse=True); b = np.zeros((5,
// vals.size)); b[np.arange(5), idx] = 1 on Float[5].
// Interpretation notes:
// - np.unique (L5 row): a plain compare-and-swap sort of a, carrying the
//   original index (bubble sort, stable: swap only on strict >, written as
//   a plain Noir `if` over the fixed pass/k loop, which is the language's
//   own select over the two branches, per L4). Adjacent compares on the
//   sorted values mark each first occurrence of a distinct value; a running
//   sum of those marks is the rank among distinct values at each sorted
//   position and, at the last position, vals.size. vals itself is never read
//   beyond .size, so only the count is materialized (L2).
// - idx is wanted in the kernel's original order, but the rank just computed
//   is known in sorted order; it is read back by an equality test against
//   the index carried through the sort: idx[i] = sum_j [carried_j == i] *
//   rank_j, the "counting by a loop of equality tests" form of L5's np.unique
//   row.
// - b[np.arange(5), idx] = 1 (L5's private-index write) on the zero matrix:
//   cell (i, c) is 1 iff c == idx_i, nothing else.
// - Output: the dynamic2 instance packs the value compact and row-major at a
//   *dynamic* row stride (n_unique, not the padded width 5): row i's one-hot
//   starts at cell i * n_unique. That packing is forced by the output format
//   alone (needed by every baseline, independent of how idx/n_unique are
//   computed), not by anything in kernel.py, so the packing arithmetic
//   itself (the cell = i * n_unique + rank search below) is copied from
//   main.nr verbatim as plumbing (L7); idx and n_unique are this file's own
//   sort-based computation.
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

    // b = np.zeros((5, vals.size)); b[np.arange(5), idx] = 1. The compact
    // dynamic2 packing (row i at stride n_unique) is output-format plumbing,
    // copied from main.nr (L7); idx and n_unique are this file's own.
    let mut out: [Field; 2 + N * N] = [0; 2 + N * N];
    out[0] = N as Field;
    out[1] = n_unique;
    for i in 0..N {
        let cell = (i as Field) * n_unique + idx[i];
        std::as_witness(cell); // one witness for the equalities below
        for t in 0..N * N {
            if (t >= i) & (t <= N * i + N - 1) {
                out[2 + t] += SCALE * ((cell == t as Field) as Field);
            }
        }
    }
    out
}
