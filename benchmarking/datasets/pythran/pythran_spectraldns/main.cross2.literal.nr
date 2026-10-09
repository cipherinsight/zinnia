// pythran_spectraldns__cross2: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize full complex arrays per expression, with no CSE or dead-write elimination.
// Real-complex multiplication is two Q32 products; multiplication by 1j forms all four exact 0/1 products without extra range checks, as optimized.

// @outputs: float[3 * 2 * POINTS]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.cross2.json.
global A1: u32 = 2; // = a_1
global A2: u32 = 5; // = a_2
global A3: u32 = 7; // = a_3
global POINTS: u32 = A1 * A2 * A3;

fn main(a: [[Field; POINTS]; 3], b: [[[Field; 2]; POINTS]; 3]) -> pub [[[Field; 2]; POINTS]; 3] {
    for k in 0..3 {
        check_inputs(a[k]);
        for p in 0..POINTS {
            check_inputs(b[k][p]);
        }
    }
    let c = cross1(a, b);
    let mut result = [[[0; 2]; POINTS]; 3];
    for k in 0..3 {
        for p in 0..POINTS {
            result[k][p] = [
                c[k][p][0] * 0 - c[k][p][1] * 1,
                c[k][p][0] * 1 + c[k][p][1] * 0,
            ];
        }
    }
    result
}
fn cross1(a: [[Field; POINTS]; 3], b: [[[Field; 2]; POINTS]; 3]) -> [[[Field; 2]; POINTS]; 3] {
    let indices = [(0, 2, 2, 1), (2, 0, 0, 2), (0, 1, 1, 0)];
    let mut c = [[[0; 2]; POINTS]; 3];
    for k in 0..3 {
        let (ak, bk, al, bl) = indices[k];
        let mut plus = [[0; 2]; POINTS];
        for p in 0..POINTS {
            for part in 0..2 {
                plus[p][part] = mul(a[ak][p], b[bk][p][part]);
            }
        }
        let mut minus = [[0; 2]; POINTS];
        for p in 0..POINTS {
            for part in 0..2 {
                minus[p][part] = mul(a[al][p], b[bl][p][part]);
            }
        }
        for p in 0..POINTS {
            for part in 0..2 {
                c[k][p][part] = plus[p][part] - minus[p][part];
            }
        }
    }
    c
}
