// npbench_weather_stencils_hdiff_hdiff__hdiff: literal translation of kernel.py (LITERAL_BASELINES.md).
// Every array operator is a separate full pass, including unused lap_field corners (L2).
// np.where computes its rounded product, comparison and select in separate passes.
// Discarded-only border inputs use shared integer limbs, retaining their actual
// quantized values without the signed64 bounds omitted by optimized. Four never-read
// outer corners and overwritten out_field are not loaded. Exact 4*x matches optimized.
use crate::prelude::*;
const I: usize = 11; // = out_field_0
const J: usize = 11; // = out_field_1
const K: usize = 10; // = out_field_2
const IN_I: usize = I + 4;
const IN_J: usize = J + 4;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input = inputs.floats("in_field", &[IN_I, IN_J, K])?;
    let coeff = inputs.floats("coeff", &[I, J, K])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let zero = q.float_const(ctx, 0.0);
    let mut field = vec![zero; IN_I * IN_J * K];
    for i in 0..IN_I {
        for j in 0..IN_J {
            if !((i == 0 || i == IN_I - 1) && (j == 0 || j == IN_J - 1)) {
                for k in 0..K {
                    let at = (i * IN_J + j) * K + k;
                    let used = (i >= 2 && i < IN_I - 2) || (j >= 2 && j < IN_J - 2)
                        || ((i > 0 && i < IN_I - 1) && (j > 0 && j < IN_J - 1));
                    field[at] = if used { q.load_float(ctx, input[at]) }
                        else { load_discarded_q32(&q, ctx, input[at]) };
                }
            }
        }
    }
    let coeff = q.load_floats(ctx, &coeff);
    let mut four = vec![zero; (I + 2) * (J + 2) * K];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                four[(((i) * (J + 2) + (j)) * K) + k] = q
                    .mul_int(
                        ctx,
                        Constant(fe(4)),
                        field[(((i + 1) * IN_J + (j + 1)) * K) + k],
                    );
            }
        }
    }
    let mut s1 = vec![zero; (I + 2) * (J + 2) * K];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                s1[(((i) * (J + 2) + (j)) * K) + k] = q.add(
                    ctx,
                    field[(((i + 2) * IN_J + (j + 1)) * K) + k],
                    field[(((i) * IN_J + (j + 1)) * K) + k],
                );
            }
        }
    }
    let mut s2 = vec![zero; (I + 2) * (J + 2) * K];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                s2[(((i) * (J + 2) + (j)) * K) + k] = q.add(
                    ctx,
                    s1[(((i) * (J + 2) + (j)) * K) + k],
                    field[(((i + 1) * IN_J + (j + 2)) * K) + k],
                );
            }
        }
    }
    let mut s3 = vec![zero; (I + 2) * (J + 2) * K];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                s3[(((i) * (J + 2) + (j)) * K) + k] = q.add(
                    ctx,
                    s2[(((i) * (J + 2) + (j)) * K) + k],
                    field[(((i + 1) * IN_J + (j)) * K) + k],
                );
            }
        }
    }
    let mut lap = vec![zero; (I + 2) * (J + 2) * K];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                lap[(((i) * (J + 2) + (j)) * K) + k] = q.sub(
                    ctx,
                    four[(((i) * (J + 2) + (j)) * K) + k],
                    s3[(((i) * (J + 2) + (j)) * K) + k],
                );
            }
        }
    }
    let mut resx = vec![zero; (I + 1) * (J) * K];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                resx[(((i) * (J) + (j)) * K) + k] = q.sub(
                    ctx,
                    lap[(((i + 1) * (J + 2) + (j + 1)) * K) + k],
                    lap[(((i) * (J + 2) + (j + 1)) * K) + k],
                );
            }
        }
    }
    let mut diffx = vec![zero; (I + 1) * (J) * K];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                diffx[(((i) * (J) + (j)) * K) + k] = q.sub(
                    ctx,
                    field[(((i + 2) * IN_J + (j + 2)) * K) + k],
                    field[(((i + 1) * IN_J + (j + 2)) * K) + k],
                );
            }
        }
    }
    let mut prodx = vec![zero; (I + 1) * (J) * K];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                prodx[(((i) * (J) + (j)) * K) + k] = q
                    .mul(
                        ctx,
                        resx[(((i) * (J) + (j)) * K) + k],
                        diffx[(((i) * (J) + (j)) * K) + k],
                    )
                    .cell;
            }
        }
    }
    let mut condx = vec![zero; (I + 1) * (J) * K];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                condx[(((i) * (J) + (j)) * K) + k] = {
                    let p = q.bound(ctx, prodx[(((i) * (J) + (j)) * K) + k]);
                    let z = q.bounded_const(ctx, 0);
                    q.gt(ctx, p, z)
                };
            }
        }
    }
    let mut flx = vec![zero; (I + 1) * (J) * K];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                flx[(((i) * (J) + (j)) * K) + k] = q.gate.select(
                    ctx,
                    zero,
                    resx[(((i) * (J) + (j)) * K) + k],
                    condx[(((i) * (J) + (j)) * K) + k],
                );
            }
        }
    }
    let mut resy = vec![zero; (I) * (J + 1) * K];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                resy[(((i) * (J + 1) + (j)) * K) + k] = q.sub(
                    ctx,
                    lap[(((i + 1) * (J + 2) + (j + 1)) * K) + k],
                    lap[(((i + 1) * (J + 2) + (j)) * K) + k],
                );
            }
        }
    }
    let mut diffy = vec![zero; (I) * (J + 1) * K];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                diffy[(((i) * (J + 1) + (j)) * K) + k] = q.sub(
                    ctx,
                    field[(((i + 2) * IN_J + (j + 2)) * K) + k],
                    field[(((i + 2) * IN_J + (j + 1)) * K) + k],
                );
            }
        }
    }
    let mut prody = vec![zero; (I) * (J + 1) * K];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                prody[(((i) * (J + 1) + (j)) * K) + k] = q
                    .mul(
                        ctx,
                        resy[(((i) * (J + 1) + (j)) * K) + k],
                        diffy[(((i) * (J + 1) + (j)) * K) + k],
                    )
                    .cell;
            }
        }
    }
    let mut condy = vec![zero; (I) * (J + 1) * K];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                condy[(((i) * (J + 1) + (j)) * K) + k] = {
                    let p = q.bound(ctx, prody[(((i) * (J + 1) + (j)) * K) + k]);
                    let z = q.bounded_const(ctx, 0);
                    q.gt(ctx, p, z)
                };
            }
        }
    }
    let mut fly = vec![zero; (I) * (J + 1) * K];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                fly[(((i) * (J + 1) + (j)) * K) + k] = q.gate.select(
                    ctx,
                    zero,
                    resy[(((i) * (J + 1) + (j)) * K) + k],
                    condy[(((i) * (J + 1) + (j)) * K) + k],
                );
            }
        }
    }
    let mut gx = vec![zero; (I) * (J) * K];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                gx[(((i) * (J) + (j)) * K) + k] = q.sub(
                    ctx,
                    flx[(((i + 1) * (J) + (j)) * K) + k],
                    flx[(((i) * (J) + (j)) * K) + k],
                );
            }
        }
    }
    let mut gx_y1 = vec![zero; (I) * (J) * K];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                gx_y1[(((i) * (J) + (j)) * K) + k] = q.add(
                    ctx,
                    gx[(((i) * (J) + (j)) * K) + k],
                    fly[(((i) * (J + 1) + (j + 1)) * K) + k],
                );
            }
        }
    }
    let mut divergence = vec![zero; (I) * (J) * K];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                divergence[(((i) * (J) + (j)) * K) + k] = q.sub(
                    ctx,
                    gx_y1[(((i) * (J) + (j)) * K) + k],
                    fly[(((i) * (J + 1) + (j)) * K) + k],
                );
            }
        }
    }
    let mut scaled = vec![zero; (I) * (J) * K];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                scaled[(((i) * (J) + (j)) * K) + k] = q
                    .mul(
                        ctx,
                        coeff[(((i) * (J) + (j)) * K) + k],
                        divergence[(((i) * (J) + (j)) * K) + k],
                    )
                    .cell;
            }
        }
    }
    let mut result = vec![zero; (I) * (J) * K];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                result[(((i) * (J) + (j)) * K) + k] = q.sub(
                    ctx,
                    field[(((i + 2) * IN_J + (j + 2)) * K) + k],
                    scaled[(((i) * (J) + (j)) * K) + k],
                );
            }
        }
    }
    out.floats(&result);
    Ok(())
}

// Statement input encoding for a value used only by discarded source work.
// Three bounded integer limbs reconstruct the full quantized i128 input without
// imposing a signed64 window on it. Only shared input loaders are used (L6).
fn load_discarded_q32(q: &Q32, ctx: &mut Ctx, value: f64) -> Cell {
    let code = quantize(value);
    let mask = (1i128 << 63) - 1;
    let low = q.load_int(ctx, (code & mask) as i64);
    let middle = q.load_int(ctx, ((code >> 63) & mask) as i64);
    let high = q.load_int(ctx, (code >> 126) as i64);
    let middle = q.mul_int(ctx, middle, Constant(fe(1i128 << 63)));
    let high = q.mul_int(ctx, high, Constant(fe(1i128 << 126)));
    let lower = q.add(ctx, low, middle);
    q.add(ctx, lower, high)
}
