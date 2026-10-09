//! pythran_periodic_dist__dist (Pythran periodic_dist), halo2-base baseline.
//! Kernel: for particles at `x, y, z: Float[N]` in a box of integer side
//! `L`, `dx[i, j] = x[j] - x[i]` (likewise `dy`, `dz`); along each axis whose
//! boolean flag is set, `dx[dx > L / 2] -= L`, then `dx[dx < -L / 2] += L`;
//! `d = sqrt(dx**2 + dy**2 + dz**2)` with `d[d == 0] = -1`. Returns
//! `(d, dx, dy, dz)`.
//! Per axis, with `h = flag * L / 2` (a Q32 value, window-checked: NumPy
//! computes `L / 2` when the flag is set), `f = flag * L` and
//! `v = c[j] - c[i]` for `i < j`, the two comparisons `a = v > h` and
//! `b = v < -h` decide both entries: the second update sees `v - L` only
//! when `a` held, and then `v - L < -h` is false, so
//! `dx[i, j] = v + f (b - a - a b)` and, as `-v > h` is `b`,
//! `dx[j, i] = -v + f (a - b - a b) = -dx[i, j] - 2 f a b`. The comparisons
//! are on `c[i] + h + 2^64` and `c[j] + 2^64` (`CMP_BITS` bits, both
//! non-negative), so `v` needs no window check of its own. The diagonal
//! (`v = 0`) is `-f [h < 0]` for every `i`, computed once, as is its `d`.
//! The squares are rounded one by one (NumPy materializes `dx**2`). With
//! `F` the Q32 code of `f` (`2^32 f`), `dx[j, i]^2 = dx[i, j]^2
//! + 4 F a b (dx[i, j] + F)`, and `4 F a b (dx[i, j] + F) / 2^32` is an
//! integer, so `floor(dx[j, i]^2 / 2^32)` is `floor(dx[i, j]^2 / 2^32)` plus
//! that integer: one rescaled product per pair, not two. `d == 0` is tested
//! on `d` itself.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0

/// Comparisons of `c[i] + h + 2^64` with `c[j] + 2^64`, both in `[0, 2^65)`.
const CMP_BITS: usize = 65;

/// One axis after the periodic updates: the `N x N` entries and their
/// rounded squares (the diagonal entries are one cell each, computed once).
struct Axis {
    entries: Vec<Cell>,
    squares: Vec<Cell>,
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let coordinates = ["x", "y", "z"].map(|name| inputs.floats(name, &[N]));
    let l = inputs.int("L")?;
    let flags = ["periodicX", "periodicY", "periodicZ"].map(|name| inputs.bools(name, &[]));
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut loaded = Vec::with_capacity(3);
    for c in coordinates {
        loaded.push(q.load_floats(ctx, &c?));
    }
    let l = q.load_int(ctx, l);
    let mut periodic = Vec::with_capacity(3);
    for f in flags {
        periodic.push(q.load_bools(ctx, &f?)[0]);
    }

    let axes: Vec<Axis> = loaded.iter().zip(&periodic).map(|(c, p)| axis(&q, ctx, c, *p, l)).collect();
    let minus_one = Constant(fe(-SCALE));
    let mut d = Vec::with_capacity(N * N);
    let mut diagonal_d: Option<Cell> = None;
    for at in 0..N * N {
        if at % (N + 1) == 0 && diagonal_d.is_some() {
            d.extend(diagonal_d);
            continue;
        }
        let sum = q.sum(ctx, axes.iter().map(|axis| axis.squares[at]));
        let root = q.sqrt(ctx, sum);
        let zero = q.gate.is_zero(ctx, root.cell);
        let dij = q.gate.select(ctx, minus_one, root.cell, zero);
        if at == 0 {
            diagonal_d = Some(dij);
        }
        d.push(dij);
    }
    out.floats(&d);
    for axis in &axes {
        out.floats(&axis.entries);
    }
    Ok(())
}

/// `c[j] - c[i]` after the periodic updates under `p`, and the squares.
fn axis(q: &Q32, ctx: &mut Ctx, c: &[Cell], p: Cell, l: Cell) -> Axis {
    let f = q.gate.mul(ctx, p, l); // flag * L
    let f_float = q.int_to_float(ctx, f); // F, its Q32 code
    let four_f = q.gate.mul(ctx, f, Constant(Fr::from(4)));
    let minus_two_f_float = q.gate.mul(ctx, f_float, Constant(-Fr::from(2)));
    let h = q.gate.mul(ctx, f, Constant(fe(SCALE / 2))); // flag * L / 2
    let h = q.bound(ctx, h);
    let two_64 = Constant(fe(1i128 << 64));
    let shifted: Vec<Cell> = c.iter().map(|ci| q.gate.add(ctx, *ci, two_64)).collect();
    let raised: Vec<Cell> = shifted.iter().map(|si| q.gate.add(ctx, *si, h.cell)).collect();
    // The diagonal: v = 0, a = b = [h < 0].
    let h_negative = q.is_neg(ctx, h);
    let minus_f = q.neg(ctx, f_float);
    let diagonal = q.gate.mul(ctx, minus_f, h_negative);
    let diagonal_square = q.mul(ctx, diagonal, diagonal).cell;
    let mut entries = vec![diagonal; N * N];
    let mut squares = vec![diagonal_square; N * N];
    for i in 0..N {
        for j in i + 1..N {
            let a = q.range.is_less_than(ctx, raised[i], shifted[j], CMP_BITS); // v > h
            let b = q.range.is_less_than(ctx, raised[j], shifted[i], CMP_BITS); // v < -h
            let ab = q.gate.mul(ctx, a, b);
            let update = q.sub(ctx, b, a);
            let update = q.sub(ctx, update, ab); // b - a - ab
            let v = q.sub(ctx, c[j], c[i]);
            let forward = q.gate.mul_add(ctx, f_float, update, v);
            // dx[j, i] = -dx[i, j] - 2 F a b
            let minus_forward = q.neg(ctx, forward);
            let backward = q.gate.mul_add(ctx, ab, minus_two_f_float, minus_forward);
            entries[i * N + j] = forward;
            entries[j * N + i] = backward;
            let square = q.mul(ctx, forward, forward).cell;
            // + 4 F a b (dx[i, j] + F) / 2^32 = 4 f a b (dx[i, j] + F)
            let gap = q.add(ctx, forward, f_float);
            let gap = q.gate.mul(ctx, gap, ab);
            squares[i * N + j] = square;
            squares[j * N + i] = q.gate.mul_add(ctx, gap, four_f, square);
        }
    }
    Axis { entries, squares }
}
