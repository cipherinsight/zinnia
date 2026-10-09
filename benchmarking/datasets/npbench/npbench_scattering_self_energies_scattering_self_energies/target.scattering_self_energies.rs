//! npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies
//! (NPBench scattering_self_energies), halo2-base baseline.
//! Kernel: for every `k, E, q, w, i, j, a, b` with `E >= w`,
//! `Sigma[k, E, a] += (G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]) @ (dH[a, b, j] * D[q, w, a, b, i, j])`
//! on complex `NORB x NORB` blocks, Sigma in place; the output is `Sigma`
//! after the call (`mutated_outputs`).
//! The two factors depend on fewer indices than the loop nest runs over, so
//! each is computed once: the gathered blocks per `(k, E', a, b)`, the left
//! factor per `(k, E', a, b, i)`, the right one per `(q, w, a, b, i, j)`.
//! Complex multiplication uses four individually rounded real products,
//! then a real subtraction and imaginary addition. Matrix products sum
//! these rounded complex products in source order, without fused rescaling.
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
                        left[left_at(k, e, a, b, i)] = complex_matmul(&q, ctx, &gathered, factor);
                    }
                }
            }
        }
    }

    // dH[a, b, j] * D[q, w, a, b, i, j], per (q, w, a, b, i, j), with the
    // same elementwise products reused by the enclosing loop.
    let mut right = vec![Vec::new(); NQZ * NW * NA * NB * N3D * N3D];
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
                            let block: Vec<Complex> = dh_block(a, b, j)
                                .iter()
                                .map(|h| complex_product(&q, ctx, *h, factor))
                                .collect();
                            right[right_at(qz, w, a, b, i, j)] = block;
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
                                    let factor = &right[right_at(qz, w, a, b, i, j)];
                                    let product =
                                        complex_matmul(&q, ctx, &left[left_at(k, e - w, a, b, i)], factor);
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

/// Four rounded products, then the source complex addition/subtraction.
fn complex_product(q: &Q32, ctx: &mut Ctx, x: Complex, y: Complex) -> Complex {
    let ac = q.mul(ctx, x[0], y[0]);
    let bd = q.mul(ctx, x[1], y[1]);
    let ad = q.mul(ctx, x[0], y[1]);
    let bc = q.mul(ctx, x[1], y[0]);
    [q.sub(ctx, ac.cell, bd.cell), q.add(ctx, ad.cell, bc.cell)]
}

/// Complex matrix product: round each scalar product before summing.
fn complex_matmul(q: &Q32, ctx: &mut Ctx, x: &[Complex], y: &[Complex]) -> Vec<Complex> {
    let mut out = Vec::with_capacity(BLOCK);
    for r in 0..NORB {
        for c in 0..NORB {
            let terms: Vec<Complex> = (0..NORB)
                .map(|t| complex_product(q, ctx, x[r * NORB + t], y[t * NORB + c])).collect();
            out.push([q.sum(ctx, terms.iter().map(|v| v[0])), q.sum(ctx, terms.iter().map(|v| v[1]))]);
        }
    }
    out
}
