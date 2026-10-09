// pythran_fbcorr__fbcorr: literal translation of kernel.py (LITERAL_BASELINES.md).
// All seven loops retain their source nesting; every scalar product is rounded before its in-place addition.

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
const _: () = assert!(
    ROWS + 1 - H <= OUT_ROWS && COLS + 1 - W <= OUT_COLS,
    "IndexError"
);

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let imgs = inputs.floats("imgs", &[NI, ROWS, COLS, C])?;
    let filters = inputs.floats("filters", &[NF, H, W, C])?;
    let output = inputs.floats("output", &[NI, NF, OUT_ROWS, OUT_COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let imgs = q.load_floats(ctx, &imgs);
    let filters = q.load_floats(ctx, &filters);
    let mut output = q.load_floats(ctx, &output);
    for ii in 0..NI {
        for rr in 0..ROWS - H + 1 {
            for cc in 0..COLS - W + 1 {
                for hh in 0..H {
                    for ww in 0..W {
                        for jj in 0..C {
                            for ff in 0..NF {
                                let imgval =
                                    imgs[((ii * ROWS + rr + hh) * COLS + cc + ww) * C + jj];
                                let filterval = filters[((ff * H + hh) * W + ww) * C + jj];
                                let product = q.mul(ctx, imgval, filterval).cell;
                                let at = ((ii * NF + ff) * OUT_ROWS + rr) * OUT_COLS + cc;
                                output[at] = q.add(ctx, output[at], product);
                            }
                        }
                    }
                }
            }
        }
    }
    out.floats(&output);
    Ok(())
}
