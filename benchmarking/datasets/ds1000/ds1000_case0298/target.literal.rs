//! DS-1000 case 0298: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Unique uses stable bubble sort, adjacent equality and equality-test compaction.
//! Dynamic-width matrix is padded internally and packed row-major for the public output.

use crate::prelude::*;

const N: usize = 5;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // vals, idx = np.unique(a, return_inverse=True)
    // Sort a ascending, carrying the original index (bubble sort, stable,
    // N * (N - 1) / 2 compare-and-swaps total).
    let mut v = q.load_floats(ctx, &a);
    let mut p: Vec<Cell> = (0..N).map(|i| q.int_const(ctx, i as i64)).collect();
    for pass in 0..N - 1 {
        for k in 0..N - 1 - pass {
            let left = q.bound(ctx, v[k]);
            let right = q.bound(ctx, v[k + 1]);
            let gt = q.gt(ctx, left, right); // swap when v[k] > v[k + 1]
            let new_vk = q.gate.select(ctx, v[k + 1], v[k], gt);
            let new_vk1 = q.gate.select(ctx, v[k], v[k + 1], gt);
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
        let eq = q.eq(ctx, v[j], v[j - 1]);
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
    // Materialize every unique value, even though only vals.size is used later.
    let _vals: Vec<Cell> = (0..N).map(|slot| {
        let mut acc = q.int_const(ctx, 0);
        for j in 0..N {
            let at = q.eq(ctx, rank_sorted[j], Constant(fe(slot as i128)));
            let keep = q.gate.and(ctx, first[j], at);
            let term = q.gate.mul(ctx, keep, v[j]);
            acc = q.add(ctx, acc, term);
        }
        acc
    }).collect();


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

    // b = zeros((a.size, vals.size)); scatter each row at its private column.
    let zero = q.float_const(ctx, 0.0);
    let one = q.float_const(ctx, 1.0);
    let mut b = vec![zero; N * N];
    for i in 0..N {
        for c in 0..N {
            let hit = q.eq(ctx, idx[i], Constant(fe(c as i128)));
            b[i * N + c] = q.gate.select(ctx, one, b[i * N + c], hit);
        }
    }
    // Public dynamic2 layout packs the padded matrix at its dynamic stride.
    let cells: Vec<Cell> = (0..N * N).map(|t| {
        let mut acc = q.float_const(ctx, 0.0);
        for m in 1..=N {
            if t < N * m {
                let hit = q.eq(ctx, n_unique, Constant(fe(m as i128)));
                let term = q.gate.mul(ctx, hit, b[(t / m) * N + t % m]);
                acc = q.add(ctx, acc, term);
            }
        }
        acc
    }).collect();
    let rows = q.int_const(ctx, N as i64);
    out.dynamic2(rows, n_unique, &cells, Kind::Float, N, N);
    Ok(())
}
