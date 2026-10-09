// pythran_lapl3d__laplacien: literal translation of kernel.py (LITERAL_BASELINES.md).
// Full array passes; integer 4 multiplication is exact; maximum scan then scalar maximum. Only unread corner inputs omitted.
use crate::prelude::*;
const H: usize = 16; // = image_0
const W: usize = 16; // = image_1
const C: usize = 3; // = image_2

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let image = inputs.floats("image", &[H, W, C])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let zero = q.float_const(ctx, 0.0);
    let mut pixels = vec![zero; H * W * C];
    for i in 0..H {
        for j in 0..W {
            if !((i == 0 || i == H - 1) && (j == 0 || j == W - 1)) {
                for k in 0..C {
                    let at = (i * W + j) * C + k;
                    pixels[at] = q.load_float(ctx, image[at]);
                }
            }
        }
    }

    let mut values = vec![zero; (H - 2) * (W - 2) * C];
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = q.mul_int(ctx, pixels[((i) * W + (j)) * C + k], Constant(fe(4)));
            }
        }
    }
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = q.sub(ctx, values[at], pixels[((i - 1) * W + (j)) * C + k]);
            }
        }
    }
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = q.sub(ctx, values[at], pixels[((i + 1) * W + (j)) * C + k]);
            }
        }
    }
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = q.sub(ctx, values[at], pixels[((i) * W + (j - 1)) * C + k]);
            }
        }
    }
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = q.sub(ctx, values[at], pixels[((i) * W + (j + 1)) * C + k]);
            }
        }
    }
    let magnitudes: Vec<Cell> = values
        .iter()
        .map(|v| {
            let v = q.bound(ctx, *v);
            q.abs(ctx, v)
        })
        .collect();
    let mut valmax = magnitudes[0];
    for v in &magnitudes[1..] {
        let a = q.bound(ctx, valmax);
        let b = q.bound(ctx, *v);
        valmax = q.max(ctx, a, b).cell;
    }
    let one = q.bounded_const(ctx, SCALE);
    let b = q.bound(ctx, valmax);
    let valmax = q.max(ctx, one, b).cell;
    let valmax = q.add(ctx, valmax, Constant(fe(quantize(1e-9))));
    let result: Vec<Cell> = magnitudes
        .iter()
        .map(|v| {
            let den = q.bound(ctx, valmax);
            q.div(ctx, *v, den).cell
        })
        .collect();
    out.floats(&result);
    Ok(())
}
