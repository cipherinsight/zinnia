// npbench_weather_stencils_hdiff_hdiff__hdiff (NPBench weather_stencils/hdiff),
// Noir baseline.
// Kernel: horizontal diffusion on in_field: Float[I + 4, J + 4, K] with
// coeff: Float[I, J, K], written into out_field: Float[I, J, K] (the output,
// `mutated_outputs`; overwritten whole before it is read, so not a parameter
// of the circuit). With lap the 5-point Laplacian of in_field on [I + 2, J + 2]
// (4.0 * x is exact in Q32), the fluxes flx, fly are differences of lap set to
// 0 where res * (difference of in_field) > 0, and
// out = in_field[2:I + 2, 2:J + 2] - coeff * (flx[1:] - flx[:-1] + fly[:, 1:]
// - fly[:, :-1]).
// Test the Q32-rounded product before comparing with zero, as in Zinnia.
// Only input elements that reach an output are domain-checked.
// @outputs: float[I * J * K]

use noir_hw::{check_inputs, mul, gt};

// Sizes (hw_check.sizes): from fixture.json.
global I: u32 = 11; // = out_field_0
global J: u32 = 11; // = out_field_1
global K: u32 = 10; // = out_field_2
global IN_I: u32 = I + 4;
global IN_J: u32 = J + 4;

// Source predicate: multiply with Q32 rounding, then compare with zero.
fn limit(res: Field, diff: Field) -> Field {
    if gt(mul(res, diff), 0) { 0 } else { res }
}

// lap[a][b] is used unless it is a corner (a in {0, I + 1} and b in {0, J + 1}).
fn lap_used(a: u32, b: u32) -> bool {
    !(((a == 0) | (a == I + 1)) & ((b == 0) | (b == J + 1)))
}

// in_field[r][c] is read by a used lap: lap[a][b] reads the five points around
// (a + 1, b + 1), and every other read point is one of those.
fn is_read(r: u32, c: u32) -> bool {
    let mut read = false;
    // (dr, dc) + 1: the offsets of the five points from (a + 1, b + 1).
    for o in 0..5 {
        let (dr, dc) = [(1, 1), (2, 1), (0, 1), (1, 2), (1, 0)][o];
        // a + 1 = r - (dr - 1), i.e. a = r + 1 - dr - 1 = r - dr.
        if (r >= dr) & (c >= dc) {
            let (a, b) = (r - dr, c - dc);
            if (a < I + 2) & (b < J + 2) {
                read |= lap_used(a, b);
            }
        }
    }
    read
}

fn lap(x: [[[Field; K]; IN_J]; IN_I], a: u32, b: u32, k: u32) -> Field {
    4 * x[a + 1][b + 1][k] - (x[a + 2][b + 1][k] + x[a][b + 1][k] + x[a + 1][b + 2][k] + x[a + 1][b][k])
}

fn main(in_field: [[[Field; K]; IN_J]; IN_I], coeff: [[[Field; K]; J]; I]) -> pub [[[Field; K]; J]; I] {
    // Input domain: every element read is a Q32 code in the window.
    for r in 0..IN_I {
        for c in 0..IN_J {
            if is_read(r, c) {
                check_inputs(in_field[r][c]);
            }
        }
    }
    for a in 0..I {
        for b in 0..J {
            check_inputs(coeff[a][b]);
        }
    }
    let mut flx: [[[Field; K]; J]; I + 1] = [[[0; K]; J]; I + 1];
    for a in 0..I + 1 {
        for b in 0..J {
            for k in 0..K {
                let res = lap(in_field, a + 1, b + 1, k) - lap(in_field, a, b + 1, k);
                let diff = in_field[a + 2][b + 2][k] - in_field[a + 1][b + 2][k];
                flx[a][b][k] = limit(res, diff);
            }
        }
    }
    let mut fly: [[[Field; K]; J + 1]; I] = [[[0; K]; J + 1]; I];
    for a in 0..I {
        for b in 0..J + 1 {
            for k in 0..K {
                let res = lap(in_field, a + 1, b + 1, k) - lap(in_field, a + 1, b, k);
                let diff = in_field[a + 2][b + 2][k] - in_field[a + 2][b + 1][k];
                fly[a][b][k] = limit(res, diff);
            }
        }
    }
    let mut out: [[[Field; K]; J]; I] = [[[0; K]; J]; I];
    for a in 0..I {
        for b in 0..J {
            for k in 0..K {
                let divergence = flx[a + 1][b][k] - flx[a][b][k] + fly[a][b + 1][k] - fly[a][b][k];
                out[a][b][k] = in_field[a + 2][b + 2][k] - mul(coeff[a][b][k], divergence);
            }
        }
    }
    out
}
