//! npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies
//! (NPBench scattering_self_energies), halo2-base baseline.
//! Kernel: for every `k, E, q, w, i, j, a, b` with `E >= w`,
//! `Sigma[k, E, a] += (G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]) @ (dH[a, b, j] * D[q, w, a, b, i, j])`
//! on complex `NORB x NORB` blocks, Sigma in place; the output is `Sigma`
//! after the call (`mutated_outputs`).
//! The two factors depend on fewer indices than the loop nest runs over, so
//! each is computed once: the gathered blocks per `(k, E', a, b)`, the left
//! factor per `(k, E', a, b, i)`, the right one per `(q, w, a, b, i, j)`.
//! Every element of a complex `@` is one reduction per part, rescaled once
//! (NPB_BASELINES.md 5.3), and so is every complex product of
//! `dH[a, b, j] * D[...]`; the products of the nest are rounded one by one,
//! as NumPy materializes each before adding it to Sigma (the adds are exact).
//! The gather is a one-hot over the `-GA..GA` valid NumPy indices
//! (`q.index_hits`), computed once per `(a, b)`; it pins `neigh_idx[a, b]` to
//! a valid index, which implies its domain (an int64), so `neigh_idx` gets no
//! separate domain check.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NA: usize = 6; // = neigh_idx_0
const NB: usize = 2; // = neigh_idx_1
const N3D: usize = 2; // = dH_2
const NORB: usize = 3; // = dH_3
const NKZ: usize = 2; // = G_0
const NE: usize = 4; // = G_1
const GA: usize = 6; // = G_2
const NQZ: usize = 2; // = D_0
const NW: usize = 2; // = D_1
const BLOCK: usize = NORB * NORB;

/// A complex value: real and imaginary part.
type Complex = [Cell; 2];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let neigh_idx = inputs.ints("neigh_idx", &[NA, NB])?;
    let dh = inputs.complexes("dH", &[NA, NB, N3D, NORB, NORB])?;
    let g = inputs.complexes("G", &[NKZ, NE, GA, NORB, NORB])?;
    let d = inputs.complexes("D", &[NQZ, NW, NA, NB, N3D, N3D])?;
    let sigma = inputs.complexes("Sigma", &[NKZ, NE, NA, NORB, NORB])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let complexes = |ctx: &mut Ctx, values: &[f64]| -> Vec<Complex> {
        q.load_floats(ctx, values).chunks(2).map(|c| [c[0], c[1]]).collect()
    };
    // neigh_idx is pinned to valid indices by `index_hits` (see the header).
    let neigh_idx: Vec<Cell> = neigh_idx.iter().map(|v| ctx.load_witness(fe(*v as i128))).collect();
    let dh = complexes(ctx, &dh);
    let g = complexes(ctx, &g);
    let d = complexes(ctx, &d);
    let sigma = complexes(ctx, &sigma);
    let dh_block = |a: usize, b: usize, i: usize| &dh[((a * NB + b) * N3D + i) * BLOCK..][..BLOCK];

    // G[k, E', neigh_idx[a, b]] @ dH[a, b, i], per (k, E', a, b, i).
    let mut left = vec![Vec::new(); NKZ * NE * NA * NB * N3D];
    let left_at = |k: usize, e: usize, a: usize, b: usize, i: usize| (((k * NE + e) * NA + a) * NB + b) * N3D + i;
    for a in 0..NA {
        for b in 0..NB {
            let hits = q.index_hits(ctx, neigh_idx[a * NB + b], GA);
            for k in 0..NKZ {
                for e in 0..NE {
                    let candidates = |p: usize| &g[((k * NE + e) * GA + p) * BLOCK..][..BLOCK];
                    let gathered: Vec<Complex> = (0..BLOCK)
                        .map(|t| {
                            [0, 1].map(|part| {
                                let column = (0..GA).map(|p| Existing(candidates(p)[t][part]));
                                q.gate.inner_product(ctx, hits.iter().map(|h| Existing(*h)), column)
                            })
                        })
                        .collect();
                    for i in 0..N3D {
                        let factor = dh_block(a, b, i);
                        let minus_im = negated_imaginary(&q, ctx, factor);
                        left[left_at(k, e, a, b, i)] = complex_matmul(&q, ctx, &gathered, factor, &minus_im);
                    }
                }
            }
        }
    }

    // dH[a, b, j] * D[q, w, a, b, i, j], per (q, w, a, b, i, j), with the
    // negated imaginary parts the products with it use.
    let mut right = vec![(Vec::new(), Vec::new()); NQZ * NW * NA * NB * N3D * N3D];
    let right_at = |qz: usize, w: usize, a: usize, b: usize, i: usize, j: usize| {
        ((((qz * NW + w) * NA + a) * NB + b) * N3D + i) * N3D + j
    };
    for qz in 0..NQZ {
        for w in 0..NW {
            for a in 0..NA {
                for b in 0..NB {
                    for i in 0..N3D {
                        for j in 0..N3D {
                            let factor = d[right_at(qz, w, a, b, i, j)];
                            let minus_im = q.gate.neg(ctx, factor[1]);
                            let block: Vec<Complex> = dh_block(a, b, j)
                                .iter()
                                .map(|h| complex_dot(&q, ctx, &[*h], &[factor], &[minus_im]))
                                .collect();
                            let block_minus_im = negated_imaginary(&q, ctx, &block);
                            right[right_at(qz, w, a, b, i, j)] = (block, block_minus_im);
                        }
                    }
                }
            }
        }
    }

    // Sigma[k, E, a] += left @ right, summed exactly at the end.
    let sigma_at = |k: usize, e: usize, a: usize| ((k * NE + e) * NA + a) * BLOCK;
    let mut terms: Vec<[Vec<Cell>; 2]> = sigma.iter().map(|s| [vec![s[0]], vec![s[1]]]).collect();
    for k in 0..NKZ {
        for e in 0..NE {
            for qz in 0..NQZ {
                for w in 0..NW.min(e + 1) {
                    for i in 0..N3D {
                        for j in 0..N3D {
                            for a in 0..NA {
                                for b in 0..NB {
                                    let (factor, minus_im) = &right[right_at(qz, w, a, b, i, j)];
                                    let product =
                                        complex_matmul(&q, ctx, &left[left_at(k, e - w, a, b, i)], factor, minus_im);
                                    for (t, value) in product.iter().enumerate() {
                                        for part in 0..2 {
                                            terms[sigma_at(k, e, a) + t][part].push(value[part]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let result: Vec<Cell> = terms.into_iter().flat_map(|parts| parts.map(|p| q.sum(ctx, p))).collect();
    out.floats(&result);
    Ok(())
}

/// `-Im v` for every element of a block.
fn negated_imaginary(q: &Q32, ctx: &mut Ctx, block: &[Complex]) -> Vec<Cell> {
    block.iter().map(|v| q.gate.neg(ctx, v[1])).collect()
}

/// `x @ y` for complex `NORB x NORB` blocks, row-major, given `-Im y`: each
/// part of each element one rescale.
fn complex_matmul(q: &Q32, ctx: &mut Ctx, x: &[Complex], y: &[Complex], minus_im: &[Cell]) -> Vec<Complex> {
    let mut out = Vec::with_capacity(BLOCK);
    for r in 0..NORB {
        for c in 0..NORB {
            let row: Vec<Complex> = (0..NORB).map(|t| x[r * NORB + t]).collect();
            let column: Vec<Complex> = (0..NORB).map(|t| y[t * NORB + c]).collect();
            let column_minus_im: Vec<Cell> = (0..NORB).map(|t| minus_im[t * NORB + c]).collect();
            out.push(complex_dot(q, ctx, &row, &column, &column_minus_im));
        }
    }
    out
}

/// `sum_t x_t y_t` for complex values, given `-Im y_t`: real part
/// `sum Re x Re y - Im x Im y`, imaginary part `sum Re x Im y + Im x Re y`,
/// each rescaled once.
fn complex_dot(q: &Q32, ctx: &mut Ctx, x: &[Complex], y: &[Complex], y_minus_im: &[Cell]) -> Complex {
    let xs = || x.iter().flat_map(|v| [Existing(v[0]), Existing(v[1])]);
    let re = q.gate.inner_product(ctx, xs(), y.iter().zip(y_minus_im).flat_map(|(v, m)| [Existing(v[0]), Existing(*m)]));
    let im = q.gate.inner_product(ctx, xs(), y.iter().flat_map(|v| [Existing(v[1]), Existing(v[0])]));
    [q.rescale(ctx, re).cell, q.rescale(ctx, im).cell]
}
