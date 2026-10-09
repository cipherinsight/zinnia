//! pythran_fbcorr__fbcorr (Pythran fbcorr), halo2-base baseline.
//! Kernel: filter-bank correlation, `output[ii, ff, rr, cc] += imgs[ii, rr +
//! hh, cc + ww, jj] * filters[ff, hh, ww, jj]` over `hh < H`, `ww < W`,
//! `jj < C`, for `rr < ROWS - H + 1`, `cc < COLS - W + 1`, with
//! `imgs: Float[NI, ROWS, COLS, C]`, `filters: Float[NF, H, W, C]`, in place
//! into `output: Float[NI, NF, OUT_ROWS, OUT_COLS]`; the output is `output`
//! after the call (`mutated_outputs`).
//! Match the source's scalar multiply-then-add recurrence: round every
//! product to Q32 before adding it. No dot-product reassociation.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 4; // = imgs_0
const ROWS: usize = 8; // = imgs_1
const COLS: usize = 8; // = imgs_2
const C: usize = 3; // = imgs_3
const NF: usize = 6; // = filters_0
const H: usize = 5; // = filters_1
const W: usize = 5; // = filters_2
const OUT_ROWS: usize = 4; // = output_2
const OUT_COLS: usize = 4; // = output_3
const _: () = assert!(ROWS + 1 - H <= OUT_ROWS && COLS + 1 - W <= OUT_COLS, "IndexError");

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let imgs = inputs.floats("imgs", &[NI, ROWS, COLS, C])?;
    let filters = inputs.floats("filters", &[NF, H, W, C])?;
    let output = inputs.floats("output", &[NI, NF, OUT_ROWS, OUT_COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let imgs = q.load_floats(ctx, &imgs);
    let filters = q.load_floats(ctx, &filters);
    let mut output = q.load_floats(ctx, &output);
    let pixel = |ii: usize, r: usize, c: usize, jj: usize| ((ii * ROWS + r) * COLS + c) * C + jj;
    let tap = |ff: usize, hh: usize, ww: usize, jj: usize| ((ff * H + hh) * W + ww) * C + jj;
    let at = |ii: usize, ff: usize, rr: usize, cc: usize| ((ii * NF + ff) * OUT_ROWS + rr) * OUT_COLS + cc;
    for ii in 0..NI {
        for rr in 0..ROWS + 1 - H {
            for cc in 0..COLS + 1 - W {
                let mut window: Vec<Cell> = Vec::with_capacity(H * W * C);
                for hh in 0..H {
                    for ww in 0..W {
                        window.extend((0..C).map(|jj| imgs[pixel(ii, rr + hh, cc + ww, jj)]));
                    }
                }
                for ff in 0..NF {
                    let filter: Vec<Cell> = (0..H * W * C).map(|t| filters[tap(ff, 0, 0, 0) + t]).collect();
                    let products: Vec<Cell> = window.iter().zip(&filter)
                        .map(|(a, b)| q.mul(ctx, *a, *b).cell).collect();
                    let sum = q.sum(ctx, products);
                    let o = at(ii, ff, rr, cc);
                    output[o] = q.add(ctx, output[o], sum);
                }
            }
        }
    }
    out.floats(&output);
    Ok(())
}
