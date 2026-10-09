//! pythran_periodic_dist__dist (Pythran periodic_dist), halo2-base baseline.
//! Kernel: for particles at `x, y, z: Float[N]` in a box of integer side
//! `L`, `dx[i, j] = x[j] - x[i]` (likewise `dy`, `dz`); along each axis whose
//! boolean flag is set, `dx[dx > L / 2] -= L`, then `dx[dx < -L / 2] += L`;
//! `d = sqrt(dx**2 + dy**2 + dz**2)` with `d[d == 0] = -1`. Returns
//! `(d, dx, dy, dz)`.
//! Per axis, with `h = flag * L / 2` (a Q32 value, window-checked: NumPy
//! computes `L / 2` when the flag is set) and `v = c[j] - c[i]` for `i < j`,
//! the two comparisons `a = v > h` and `b = v < -h` decide both entries:
//! the second update sees `v - L` only when `a` held, and then `v - L < -h`
//! is false, so `dx[i, j] = v + flag L (b - a - a b)` and, as `-v > h` is
//! `b`, `dx[j, i] = -v + flag L (a - b - a b)`. The comparisons are on
//! `c[i] + h + 2^64` and `c[j] + 2^64` (`CMP_BITS` bits, both non-negative),
//! so `v` needs no window check of its own. The diagonal (`v = 0`) is
//! `-flag L [h < 0]` for every `i`, computed once, as is its `d`.
//! The squares are rounded one by one (NumPy materializes `dx**2`), and
//! `d == 0` is tested on `d` itself.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0

/// Comparisons of `c[i] + h + 2^64` with `c[j] + 2^64`, both in `[0, 2^65)`.
const CMP_BITS: usize = 65;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let coordinates = ["x", "y", "z"].map(|name| inputs.floats(name, &[N]));
    let l = inputs.int("L")?;
    let flags = ["periodicX", "periodicY", "periodicZ"].map(|name| inputs.bools(name, &[]));
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut axes = Vec::with_capacity(3);
    for c in coordinates {
        axes.push(q.load_floats(ctx, &c?));
    }
    let l = q.load_int(ctx, l);
    let mut periodic = Vec::with_capacity(3);
    for f in flags {
        periodic.push(q.load_bools(ctx, &f?)[0]);
    }

    let deltas: Vec<Vec<Cell>> = axes.iter().zip(&periodic).map(|(c, p)| axis_deltas(&q, ctx, c, *p, l)).collect();
    let minus_one = Constant(fe(-SCALE));
    // d at (i, j), from the squared entries of the three axes there.
    let distance_at = |ctx: &mut Ctx, at: usize| {
        let squares: Vec<Cell> = deltas.iter().map(|axis| q.mul(ctx, axis[at], axis[at]).cell).collect();
        let sum = q.sum(ctx, squares);
        let root = q.sqrt(ctx, sum);
        let zero = q.gate.is_zero(ctx, root.cell);
        q.gate.select(ctx, minus_one, root.cell, zero)
    };
    let diagonal_d = distance_at(ctx, 0);
    let mut d = Vec::with_capacity(N * N);
    for at in 0..N * N {
        d.push(if at % (N + 1) == 0 { diagonal_d } else { distance_at(ctx, at) });
    }
    out.floats(&d);
    for axis in &deltas {
        out.floats(axis);
    }
    Ok(())
}

/// The `N x N` matrix `c[j] - c[i]` after the periodic updates under `p`.
/// The diagonal entries are one cell, computed once.
fn axis_deltas(q: &Q32, ctx: &mut Ctx, c: &[Cell], p: Cell, l: Cell) -> Vec<Cell> {
    let pl = q.gate.mul(ctx, p, l);
    let l_float = q.int_to_float(ctx, pl); // flag * L as a float
    let h = q.gate.mul(ctx, pl, Constant(fe(SCALE / 2))); // flag * L / 2
    let h = q.bound(ctx, h);
    let two_64 = Constant(fe(1i128 << 64));
    let shifted: Vec<Cell> = c.iter().map(|ci| q.gate.add(ctx, *ci, two_64)).collect();
    let raised: Vec<Cell> = shifted.iter().map(|si| q.gate.add(ctx, *si, h.cell)).collect();
    // The diagonal: v = 0, a = b = [h < 0].
    let h_negative = q.is_neg(ctx, h);
    let minus_l = q.neg(ctx, l_float);
    let diagonal = q.gate.mul(ctx, minus_l, h_negative);
    let mut e = vec![diagonal; N * N];
    for i in 0..N {
        for j in i + 1..N {
            let a = q.range.is_less_than(ctx, raised[i], shifted[j], CMP_BITS); // v > h
            let b = q.range.is_less_than(ctx, raised[j], shifted[i], CMP_BITS); // v < -h
            let ab = q.gate.mul(ctx, a, b);
            let forward = q.sub(ctx, b, a);
            let forward = q.sub(ctx, forward, ab); // b - a - ab
            let backward = q.sub(ctx, a, b);
            let backward = q.sub(ctx, backward, ab); // a - b - ab
            let v = q.sub(ctx, c[j], c[i]);
            let minus_v = q.sub(ctx, c[i], c[j]);
            e[i * N + j] = q.gate.mul_add(ctx, l_float, forward, v);
            e[j * N + i] = q.gate.mul_add(ctx, l_float, backward, minus_v);
        }
    }
    e
}
