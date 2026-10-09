// pythran_fbcorr__fbcorr (Pythran fbcorr), Noir baseline.
// Kernel: filter-bank correlation, output[ii, ff, rr, cc] += imgs[ii, rr + hh,
// cc + ww, jj] * filters[ff, hh, ww, jj] over hh < H, ww < W, jj < C, for
// rr < ROWS - H + 1, cc < COLS - W + 1, with imgs: Float[NI, ROWS, COLS, C],
// filters: Float[NF, H, W, C], in place into
// output: Float[NI, NF, OUT_ROWS, OUT_COLS]; the output is `output` after the
// call (`mutated_outputs`).
// Match the source recurrence: round each product before accumulation.
// No dot-product reassociation.
// @outputs: float[NI * NF * OUT_ROWS * OUT_COLS]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global NI: u32 = 4; // = imgs_0
global ROWS: u32 = 8; // = imgs_1
global COLS: u32 = 8; // = imgs_2
global C: u32 = 3; // = imgs_3
global NF: u32 = 6; // = filters_0
global H: u32 = 5; // = filters_1
global W: u32 = 5; // = filters_2
global OUT_ROWS: u32 = 4; // = output_2
global OUT_COLS: u32 = 4; // = output_3

fn main(
    imgs: [[[[Field; C]; COLS]; ROWS]; NI],
    filters: [[[[Field; C]; W]; H]; NF],
    output: [[[[Field; OUT_COLS]; OUT_ROWS]; NF]; NI],
) -> pub [[[[Field; OUT_COLS]; OUT_ROWS]; NF]; NI] {
    // Input domain: every input element is a Q32 code in the window.
    for ii in 0..NI {
        for r in 0..ROWS {
            for c in 0..COLS {
                check_inputs(imgs[ii][r][c]);
            }
        }
        for ff in 0..NF {
            for rr in 0..OUT_ROWS {
                check_inputs(output[ii][ff][rr]);
            }
        }
    }
    for ff in 0..NF {
        for hh in 0..H {
            for ww in 0..W {
                check_inputs(filters[ff][hh][ww]);
            }
        }
    }
    let mut result = output;
    for ii in 0..NI {
        for rr in 0..ROWS + 1 - H {
            for cc in 0..COLS + 1 - W {
                for ff in 0..NF {
                    let mut acc: Field = 0;
                    for hh in 0..H {
                        for ww in 0..W {
                            for jj in 0..C {
                                acc += mul(imgs[ii][rr + hh][cc + ww][jj], filters[ff][hh][ww][jj]);
                            }
                        }
                    }
                    result[ii][ff][rr][cc] += acc;
                }
            }
        }
    }
    result
}
