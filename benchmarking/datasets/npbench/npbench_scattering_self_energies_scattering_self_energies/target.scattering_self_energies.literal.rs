// npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies: literal translation of kernel.py (LITERAL_BASELINES.md).
// All eight source loops and three statements retained; no factor/gather hoisting. Complex matmul uses current optimized arithmetic: four rounded real products per term, then running sums (fairness contract).

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
        q.load_floats(ctx, values)
            .chunks(2)
            .map(|c| [c[0], c[1]])
            .collect()
    };
    let neigh_idx = q.load_ints(ctx, &neigh_idx);
    let dh = complexes(ctx, &dh);
    let g = complexes(ctx, &g);
    let d = complexes(ctx, &d);
    let mut sigma = complexes(ctx, &sigma);
    for k in 0..NKZ {
        for e in 0..NE {
            for qz in 0..NQZ {
                for w in 0..NW {
                    for i in 0..N3D {
                        for j in 0..N3D {
                            for a in 0..NA {
                                for b in 0..NB {
                                    if e >= w {
                                        let mut gathered = Vec::with_capacity(BLOCK);
                                        for t in 0..BLOCK {
                                            let re: Vec<Cell> = (0..GA)
                                                .map(|p| {
                                                    g[((k * NE + e - w) * GA + p) * BLOCK + t][0]
                                                })
                                                .collect();
                                            let im: Vec<Cell> = (0..GA)
                                                .map(|p| {
                                                    g[((k * NE + e - w) * GA + p) * BLOCK + t][1]
                                                })
                                                .collect();
                                            gathered.push([
                                                q.index(ctx, &re, neigh_idx[a * NB + b]),
                                                q.index(ctx, &im, neigh_idx[a * NB + b]),
                                            ]);
                                        }
                                        let start_i = ((a * NB + b) * N3D + i) * BLOCK;
                                        let dhg = matmul(
                                            &q,
                                            ctx,
                                            &gathered,
                                            &dh[start_i..start_i + BLOCK],
                                        );
                                        let start_j = ((a * NB + b) * N3D + j) * BLOCK;
                                        let factor = d[((((qz * NW + w) * NA + a) * NB + b) * N3D
                                            + i)
                                            * N3D
                                            + j];
                                        let dhd: Vec<Complex> = (0..BLOCK)
                                            .map(|t| product(&q, ctx, dh[start_j + t], factor))
                                            .collect();
                                        let update = matmul(&q, ctx, &dhg, &dhd);
                                        for t in 0..BLOCK {
                                            for part in 0..2 {
                                                let pos = ((k * NE + e) * NA + a) * BLOCK + t;
                                                sigma[pos][part] =
                                                    q.add(ctx, sigma[pos][part], update[t][part]);
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
    }
    let result: Vec<Cell> = sigma.into_iter().flatten().collect();
    out.floats(&result);
    Ok(())
}
fn product(q: &Q32, ctx: &mut Ctx, x: Complex, y: Complex) -> Complex {
    let ac = q.mul(ctx, x[0], y[0]).cell;
    let bd = q.mul(ctx, x[1], y[1]).cell;
    let ad = q.mul(ctx, x[0], y[1]).cell;
    let bc = q.mul(ctx, x[1], y[0]).cell;
    [q.sub(ctx, ac, bd), q.add(ctx, ad, bc)]
}
fn matmul(q: &Q32, ctx: &mut Ctx, x: &[Complex], y: &[Complex]) -> Vec<Complex> {
    let mut result = Vec::with_capacity(BLOCK);
    for r in 0..NORB {
        for c in 0..NORB {
            let zero = q.float_const(ctx, 0.0);
            let mut acc = [zero, zero];
            for t in 0..NORB {
                let term = product(q, ctx, x[r * NORB + t], y[t * NORB + c]);
                acc = [q.add(ctx, acc[0], term[0]), q.add(ctx, acc[1], term[1])];
            }
            result.push(acc);
        }
    }
    result
}
