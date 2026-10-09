//! pythran_harris__harris: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Compute full differences before slicing, including discarded boundary values.
//! Section 3: encode the discarded-only I[0,0] through shared integer input limbs,
//! matching the optimized absence of a Q32 window without replacing its value.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 4; // = I_0
const N: usize = 6; // = I_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let image = inputs.floats("I", &[M, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let pixels: Vec<Cell> = image.iter().enumerate().map(|(i, value)| {
        if i == 0 { load_discarded_q32(&q, ctx, *value) } else { q.load_float(ctx, *value) }
    }).collect();
    let mut horizontal = Vec::new();
    for i in 0..M - 1 {
        for j in 0..N {
            horizontal.push(q.sub(ctx, pixels[(i + 1) * N + j], pixels[i * N + j]));
        }
    }
    let dx: Vec<Cell> = (0..(M - 1) * (N - 1))
        .map(|p| horizontal[(p / (N - 1)) * N + p % (N - 1) + 1])
        .collect();
    let mut vertical = Vec::new();
    for i in 0..M {
        for j in 0..N - 1 {
            vertical.push(q.sub(ctx, pixels[i * N + j + 1], pixels[i * N + j]));
        }
    }
    let dy = &vertical[N - 1..];
    let a: Vec<Cell> = dx.iter().map(|v| q.mul(ctx, *v, *v).cell).collect();
    let b: Vec<Cell> = dy.iter().map(|v| q.mul(ctx, *v, *v).cell).collect();
    let c: Vec<Cell> = (0..dx.len())
        .map(|i| q.mul(ctx, dx[i], dy[i]).cell)
        .collect();
    let tr: Vec<Cell> = (0..dx.len()).map(|i| q.add(ctx, a[i], b[i])).collect();
    let ab: Vec<Cell> = (0..dx.len()).map(|i| q.mul(ctx, a[i], b[i]).cell).collect();
    let cc: Vec<Cell> = c.iter().map(|v| q.mul(ctx, *v, *v).cell).collect();
    let det: Vec<Cell> = (0..dx.len()).map(|i| q.sub(ctx, ab[i], cc[i])).collect();
    let ktr: Vec<Cell> = tr
        .iter()
        .map(|v| q.mul(ctx, Constant(fe(quantize(0.05))), *v).cell)
        .collect();
    let ktrtr: Vec<Cell> = (0..dx.len())
        .map(|i| q.mul(ctx, ktr[i], tr[i]).cell)
        .collect();
    let result: Vec<Cell> = (0..dx.len())
        .map(|i| q.sub(ctx, det[i], ktrtr[i]))
        .collect();
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
