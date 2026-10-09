//! pythran_d2q9_nxnyns__one_time_step (Pythran d2q9_nxnyns), halo2-base baseline.
//! Kernel: one lattice-Boltzmann D2Q9 step on f1: Float[NX, NY, 9] into
//! f2: Float[MX, MY, 9]; returns `f2`. `periodic_bc(f1)` copies rows `NX - 2`
//! and 1 into rows 0 and `NX - 1`, then columns `NY - 2` and 1 into columns 0
//! and `NY - 1`; for every interior site `(i, j)` the nine populations
//! streamed in (`getf`) go to moments (`f2m_loc`), are relaxed
//! (`relaxation_loc`), go back (`m2f_loc`) and are stored at `f2[j, i]`
//! (`setf`, transposed). The 9 is D2Q9's, not a size.
//! `periodic_bc` only copies, so `f1` after it is the input read through
//! `wrap` on both axes, and every interior site reads only interior elements
//! of the input: those are the `f1` elements loaded. The written `f2`
//! elements are not loaded; the others are output as loaded.
//! The moments are exact integer combinations. Every product of `relaxation`
//! and `m2f_loc` is rounded as NumPy evaluates it (`3.0 * m1 * m1` is
//! `(3 m1) m1`, exact then rounded; `1.5 * m1 * m1` rounds both); products
//! `m2f_loc` repeats across the nine outputs (`c1 * m[0]`, ...) are computed
//! once. Constants that are powers of two times an integer (`0.25 * m`,
//! `1.5 * m`) round by `q.floor_div_const` (`floor(m / 4)`, `floor(3 m / 2)`):
//! the same values as a rescale, with a remainder of 1 or 2 bits.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NX: usize = 16; // = f1_0
const NY: usize = 16; // = f1_1
const MX: usize = 16; // = f2_0
const MY: usize = 16; // = f2_1

/// Lattice offsets of the nine populations `getf` reads (`f[i + DI, j + DJ]`).
const DI: [isize; 9] = [0, -1, 0, 1, 0, -1, 1, 1, -1];
const DJ: [isize; 9] = [0, 0, -1, 0, 1, -1, -1, 1, 1];

/// `f2m_loc`: `m[r] = sum_k MOMENTS[r][k] * f[k]`.
const MOMENTS: [[i64; 9]; 9] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1],
    [-4, -1, -1, -1, -1, 2, 2, 2, 2],
    [4, -2, -2, -2, -2, 1, 1, 1, 1],
    [0, -2, 0, 2, 0, 1, -1, -1, 1],
    [0, 0, -2, 0, 2, 1, 1, -1, -1],
    [0, 1, -1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 1, -1],
];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let f1 = inputs.floats("f1", &[NX, NY, 9])?;
    let f2 = inputs.floats("f2", &[MX, MY, 9])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // The interior of f1; `periodic_bc` maps every row and column to it.
    let mut interior: Vec<Option<Cell>> = vec![None; NX * NY * 9];
    for r in 1..NX - 1 {
        for c in 1..NY - 1 {
            for k in 0..9 {
                let at = (r * NY + c) * 9 + k;
                interior[at] = Some(q.load_float(ctx, f1[at]));
            }
        }
    }
    let wrap = |v: usize, len: usize| {
        if v == 0 {
            len - 2
        } else if v == len - 1 {
            1
        } else {
            v
        }
    };
    let after_bc = |r: usize, c: usize, k: usize| interior[(wrap(r, NX) * NY + wrap(c, NY)) * 9 + k].expect("interior");

    let mut result: Vec<Option<Cell>> = vec![None; MX * MY * 9];
    for i in 1..NX - 1 {
        for j in 1..NY - 1 {
            let streamed: Vec<Cell> =
                (0..9).map(|k| after_bc(i.wrapping_add_signed(DI[k]), j.wrapping_add_signed(DJ[k]), k)).collect();
            let m: Vec<Cell> = MOMENTS.iter().map(|row| combine(&q, ctx, row, &streamed)).collect();
            let m = relaxation(&q, ctx, &m);
            for (k, value) in moments_to_populations(&q, ctx, &m).into_iter().enumerate() {
                result[(j * MY + i) * 9 + k] = Some(value);
            }
        }
    }
    // Elements of f2 that no site writes are copies of the input.
    let result: Vec<Cell> = result
        .into_iter()
        .zip(&f2)
        .map(|(written, input)| written.unwrap_or_else(|| q.load_float(ctx, *input)))
        .collect();
    out.floats(&result);
    Ok(())
}

/// `sum_k weights[k] * values[k]` for integer weights (exact).
fn combine(q: &Q32, ctx: &mut Ctx, weights: &[i64], values: &[Cell]) -> Cell {
    let (cells, constants): (Vec<_>, Vec<_>) = weights
        .iter()
        .zip(values)
        .filter(|(w, _)| **w != 0)
        .map(|(w, v)| (Existing(*v), Constant(fe(*w as i128))))
        .unzip();
    q.gate.inner_product(ctx, cells, constants)
}

/// A float constant times a value, rounded (`floor(c * v / 2^32)`).
fn scale(q: &Q32, ctx: &mut Ctx, c: f64, v: Cell) -> Cell {
    q.mul(ctx, Constant(fe(quantize(c))), v).cell
}

/// `relaxation_loc`: the moments after relaxation.
fn relaxation(q: &Q32, ctx: &mut Ctx, m: &[Cell]) -> Vec<Cell> {
    const OMEGA_E: f64 = 1.1312217194570136;
    const OMEGA_XY: f64 = 1.8573551263001487;
    let m1_sq = q.gate.mul(ctx, m[1], m[1]);
    let m2_sq = q.gate.mul(ctx, m[2], m[2]);
    // 3.0 * m1 * m1 = (3 m1) m1: one rounding of 3 m1^2.
    let three_m1_m1 = q.gate.mul(ctx, m1_sq, Constant(fe(3)));
    let three_m1_m1 = q.rescale(ctx, three_m1_m1).cell;
    let three_m2_m2 = q.gate.mul(ctx, m2_sq, Constant(fe(3)));
    let three_m2_m2 = q.rescale(ctx, three_m2_m2).cell;
    // 1.5 * m1 * m1 = floor(3 m1 / 2) * m1, rounded.
    let halves = |ctx: &mut Ctx, v: Cell| {
        let three_v = q.gate.mul(ctx, v, Constant(fe(3)));
        q.floor_div_const(ctx, three_v, 2).0.cell
    };
    let one_half_m1 = halves(ctx, m[1]);
    let one_half_m1_m1 = q.mul(ctx, one_half_m1, m[1]).cell;
    let one_half_m2 = halves(ctx, m[2]);
    let one_half_m2_m2 = q.mul(ctx, one_half_m2, m[2]).cell;
    let m1_m1 = q.rescale(ctx, m1_sq).cell;
    let m2_m2 = q.rescale(ctx, m2_sq).cell;
    let m1_m2 = q.mul(ctx, m[1], m[2]).cell;

    let relax = |ctx: &mut Ctx, omega: f64, own: Cell, weights: &[i64], terms: &[Cell]| {
        let gap = combine(q, ctx, weights, terms);
        let step = scale(q, ctx, omega, gap);
        q.add(ctx, own, step)
    };
    let mut out = m.to_vec();
    out[3] = relax(ctx, OMEGA_E, m[3], &[-2, 1, 1, -1], &[m[0], three_m1_m1, three_m2_m2, m[3]]);
    out[4] = relax(ctx, OMEGA_E, m[4], &[1, 1, 1, -1], &[m[0], one_half_m1_m1, one_half_m2_m2, m[4]]);
    out[5] = relax(ctx, OMEGA_E, m[5], &[-1, -1], &[m[1], m[5]]);
    out[6] = relax(ctx, OMEGA_E, m[6], &[-1, -1], &[m[2], m[6]]);
    out[7] = relax(ctx, OMEGA_XY, m[7], &[1, -1, -1], &[m1_m1, m2_m2, m[7]]);
    out[8] = relax(ctx, OMEGA_XY, m[8], &[1, -1], &[m1_m2, m[8]]);
    out
}

/// `m2f_loc`: the nine populations of the relaxed moments. Each distinct
/// product is computed once.
fn moments_to_populations(q: &Q32, ctx: &mut Ctx, m: &[Cell]) -> Vec<Cell> {
    let (c0, c1, c2, c3, c4) = (1.0 / 6.0, 1.0 / 9.0, 1.0 / 18.0, 1.0 / 36.0, 1.0 / 12.0);
    let quarter = |ctx: &mut Ctx, v: Cell| q.floor_div_const(ctx, v, 4).0.cell; // 0.25 * v
    let c1_m0 = scale(q, ctx, c1, m[0]);
    let c1_m3 = scale(q, ctx, c1, m[3]);
    let c1_m4 = scale(q, ctx, c1, m[4]);
    let c0_m1 = scale(q, ctx, c0, m[1]);
    let c0_m2 = scale(q, ctx, c0, m[2]);
    let c3_m3 = scale(q, ctx, c3, m[3]);
    let c2_m4 = scale(q, ctx, c2, m[4]);
    let c0_m5 = scale(q, ctx, c0, m[5]);
    let c0_m6 = scale(q, ctx, c0, m[6]);
    let q_m7 = quarter(ctx, m[7]);
    let c2_m3 = scale(q, ctx, c2, m[3]);
    let c3_m4 = scale(q, ctx, c3, m[4]);
    let c4_m5 = scale(q, ctx, c4, m[5]);
    let c4_m6 = scale(q, ctx, c4, m[6]);
    let q_m8 = quarter(ctx, m[8]);
    let terms = [c1_m0, c0_m1, c0_m2, c3_m3, c2_m4, c0_m5, c0_m6, q_m7, c2_m3, c3_m4, c4_m5, c4_m6, q_m8, c1_m3, c1_m4];
    // Signs of `terms` in each population, in the order above.
    const SIGNS: [[i64; 15]; 9] = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
        [1, 1, 0, -1, -1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, -1, -1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, -1, -1, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [1, -1, 1, 0, 0, 0, 0, 0, 1, 1, -1, 1, -1, 0, 0],
        [1, -1, -1, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 0, 0],
        [1, 1, -1, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, 0, 0],
    ];
    SIGNS.iter().map(|signs| combine(q, ctx, signs, &terms)).collect()
}
