//! DS-1000 case 0298, halo2-base baseline.
//! Kernel: `vals, idx = np.unique(a, return_inverse=True); b = np.zeros((5,
//! vals.size)); b[np.arange(5), idx] = 1` on `Float[5]`: the one-hot rows of
//! each value's rank among the distinct values (NumPy's `unique` is sorted).
//! Output: shape (5, number of distinct values) as `dynamic2`, the cells
//! compact and row-major, zero-padded to 5 x 5.
//! The rank of `a_i` is the number of distinct values below it, counting each
//! distinct value at its first occurrence. Row `i` of `b` starts at cell
//! `i * n_unique`, so cell `t` is 1.0 when `rank_i = t - i * n_unique` for
//! some `i`: with the one-hots of `n_unique` (values 1..5) and of each rank
//! (0..4), cell `t` is `sum_m [n_unique = m] [rank_{t div m} = t mod m]`.

use crate::prelude::*;
use halo2_base::QuantumCell;

const N: usize = 5;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let ab = q.load_floats_bounded(ctx, &a);
    let a: Vec<Cell> = ab.iter().map(|x| x.cell).collect();
    // lt[i][j] = [a_i < a_j], eq[i][j] = [a_i == a_j] for i < j.
    let mut less = vec![vec![None; N]; N];
    let mut same = vec![vec![None; N]; N];
    for i in 0..N {
        for j in i + 1..N {
            less[i][j] = Some(q.lt(ctx, ab[i], ab[j]));
            same[i][j] = Some(q.eq(ctx, a[i], a[j]));
        }
    }
    let lt = |i: usize, j: usize| less[i][j].expect("i < j");
    let eq = |i: usize, j: usize| same[i][j].expect("i < j");

    // first[j] = [a_j differs from every earlier value]
    let mut first: Vec<QuantumCell<Fr>> = vec![Constant(Fr::ONE)];
    for j in 1..N {
        let mut acc = q.gate.not(ctx, eq(0, j));
        for i in 1..j {
            acc = q.gate.mul_not(ctx, eq(i, j), acc);
        }
        first.push(Existing(acc));
    }
    let n_unique = q.gate.sum(ctx, first.iter().copied());
    // rank[i] = sum_{j < i} first_j [a_j < a_i] + sum_{j > i} first_j [a_j < a_i];
    // for j > i, a first occurrence a_j differs from a_i, so [a_j < a_i] = 1 - [a_i < a_j].
    let rank: Vec<Cell> = (0..N)
        .map(|i| {
            let below: Vec<Cell> = (0..N)
                .filter(|j| *j != i)
                .map(|j| {
                    if j < i {
                        q.gate.mul(ctx, first[j], lt(j, i))
                    } else {
                        q.gate.mul_not(ctx, lt(i, j), first[j])
                    }
                })
                .collect();
            q.sum(ctx, below)
        })
        .collect();

    // cols_is[m - 1] = 1.0 * [n_unique == m]; rank_is[i][c] = [rank_i == c]
    let cols_is: Vec<Cell> = (1..=N)
        .map(|m| {
            let hit = q.eq(ctx, n_unique, Constant(fe(m as i128)));
            q.gate.mul(ctx, hit, Constant(fe(SCALE)))
        })
        .collect();
    let rank_is: Vec<Vec<Cell>> =
        rank.iter().map(|r| (0..N).map(|c| q.eq(ctx, *r, Constant(fe(c as i128)))).collect()).collect();
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
