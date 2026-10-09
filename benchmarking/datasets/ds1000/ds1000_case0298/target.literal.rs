//! DS-1000 case 0298, halo2-base baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Kernel: `vals, idx = np.unique(a, return_inverse=True); b = np.zeros((a.size,
//! vals.size)); b[np.arange(a.size), idx] = 1` on `Float[5]`.
//! Interpretation notes:
//! - `np.unique` (L5 row): a plain compare-and-swap sort of `a`, carrying the
//!   original index (bubble sort, stable: swap only on strict `>`). The
//!   sorted array's adjacent compares mark each first occurrence of a
//!   distinct value; a running sum of those marks is each sorted position's
//!   rank among distinct values and, at the last position, `vals.size`.
//!   `vals` itself (the sorted distinct values) is never read by the kernel
//!   beyond `.size`, so only the count is materialized (L2's "values the
//!   source never computes are not computed either").
//! - `idx` is in the kernel's original order, but the rank just computed is
//!   known in sorted order; it is read back by an equality test against the
//!   index carried through the sort: `idx[i] = sum_j [carried_j == i] * rank_j`.
//! - INTERPRETATION (flagged for review): the input array is loaded once
//!   with `q.load_floats_bounded`, its window proof threaded through every
//!   sort swap by `select_i64` and reused by every comparison on that
//!   element. L6 names `q.load_*_bounded` among the forbidden gadgets, but
//!   L7 carves the once-per-element domain check out of the anti-reuse rule,
//!   and CONVENTIONS.md 1.6 requires exactly one domain check per input
//!   element (a mock run's `redundant_input_checks` must be 0): calling
//!   plain `q.load_floats` and then a fresh `q.bound` at each of the sort's
//!   O(N^2) comparisons would recheck each element's domain many times over,
//!   which both wastes gates (contrary to the no-deliberately-wasteful rule)
//!   and would itself violate 1.6. Reading L6's parenthetical "(and any
//!   reuse of an I64 from one source operation in another)" as aimed at
//!   reusing a proof computed for one intermediate value in an unrelated
//!   later computation (which L3 already forbids as CSE), not at the single
//!   input-domain proof every array element needs once, this file loads
//!   every float array with the bounded loader exactly at its one read site
//!   and nowhere else.
//! - `b[np.arange(5), idx] = 1` (L5's private-index write) on the zero
//!   matrix: cell `(i, c)` is `1` iff `c == idx_i`, nothing else.
//! Output: shape (5, number of distinct values) as `dynamic2`. This instance
//! layout packs the value compact and row-major at a *dynamic* row stride
//! (`n_unique`, not the padded max width): row `i`'s one-hot starts at cell
//! `i * n_unique`. That packing is forced by the output format alone (every
//! baseline needs it, independent of how `idx`/`n_unique` are computed), not
//! by anything in `kernel.py`, so the packing arithmetic itself (the
//! `cols_is` / `rank_is` one-hot-of-the-stride construction below) is copied
//! from `target.rs` verbatim as plumbing (L7); the values it packs, `rank`
//! (= `idx`) and `n_unique`, are this file's own sort-based computation.

use crate::prelude::*;

const N: usize = 5;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // vals, idx = np.unique(a, return_inverse=True)
    // Sort a ascending, carrying the original index (bubble sort, stable,
    // N * (N - 1) / 2 compare-and-swaps total).
    let mut v: Vec<I64> = q.load_floats_bounded(ctx, &a);
    let mut p: Vec<Cell> = (0..N).map(|i| q.int_const(ctx, i as i64)).collect();
    for pass in 0..N - 1 {
        for k in 0..N - 1 - pass {
            let gt = q.gt(ctx, v[k], v[k + 1]); // swap when v[k] > v[k + 1]
            let new_vk = q.select_i64(ctx, v[k + 1], v[k], gt);
            let new_vk1 = q.select_i64(ctx, v[k], v[k + 1], gt);
            let new_pk = q.gate.select(ctx, p[k + 1], p[k], gt);
            let new_pk1 = q.gate.select(ctx, p[k], p[k + 1], gt);
            v[k] = new_vk;
            v[k + 1] = new_vk1;
            p[k] = new_pk;
            p[k + 1] = new_pk1;
        }
    }

    // first[j] = 1 when sorted position j starts a new distinct value.
    let mut first: Vec<Cell> = Vec::with_capacity(N);
    first.push(q.int_const(ctx, 1));
    for j in 1..N {
        let eq = q.eq(ctx, v[j].cell, v[j - 1].cell);
        first.push(q.gate.not(ctx, eq));
    }
    // rank_sorted[j] = (running count of distinct values through position j) - 1.
    let mut rank_sorted: Vec<Cell> = Vec::with_capacity(N);
    let mut running = first[0];
    rank_sorted.push(q.gate.sub(ctx, running, Constant(Fr::ONE)));
    for j in 1..N {
        running = q.gate.add(ctx, running, first[j]);
        rank_sorted.push(q.gate.sub(ctx, running, Constant(Fr::ONE)));
    }
    let n_unique = running; // vals.size

    // idx[i] (original order) = rank_sorted[j] for the sorted position j
    // carrying original index i.
    let idx: Vec<Cell> = (0..N)
        .map(|i| {
            let terms: Vec<Cell> = (0..N)
                .map(|j| {
                    let hit = q.eq(ctx, p[j], Constant(fe(i as i128)));
                    q.gate.mul(ctx, hit, rank_sorted[j])
                })
                .collect();
            q.sum(ctx, terms)
        })
        .collect();

    // b = np.zeros((5, vals.size)); b[np.arange(5), idx] = 1. The compact
    // dynamic2 packing (row i at stride n_unique) is output-format plumbing,
    // copied from target.rs (L7); idx and n_unique are this file's own.
    let cols_is: Vec<Cell> = (1..=N)
        .map(|m| {
            let hit = q.eq(ctx, n_unique, Constant(fe(m as i128)));
            q.gate.mul(ctx, hit, Constant(fe(SCALE)))
        })
        .collect();
    let rank_is: Vec<Vec<Cell>> =
        idx.iter().map(|r| (0..N).map(|c| q.eq(ctx, *r, Constant(fe(c as i128)))).collect()).collect();
    let cells: Vec<Cell> = (0..N * N)
        .map(|t| {
            let (hits, cols): (Vec<_>, Vec<_>) = (1..=N)
                .filter(|m| t < N * m)
                .map(|m| (Existing(rank_is[t / m][t % m]), Existing(cols_is[m - 1])))
                .unzip();
            q.gate.inner_product(ctx, hits, cols)
        })
        .collect();
    let rows = q.int_const(ctx, N as i64);
    out.dynamic2(rows, n_unique, &cells, Kind::Float, N, N);
    Ok(())
}
