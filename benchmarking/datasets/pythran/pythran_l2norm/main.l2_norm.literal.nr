// pythran_l2norm__l2_norm: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize abs, squares, sums, then roots; retain the optimized root-domain guard.

// @outputs: float[N]

use noir_hw::{check_inputs, mul, sqrt};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0
global COLS: u32 = 16; // = x_1

fn main(x: [[Field; COLS]; N]) -> pub [Field; N] {
    for i in 0..N {
        check_inputs(x[i]);
    }
    let mut magnitude = [[0; COLS]; N];
    for i in 0..N {
        for j in 0..COLS {
            magnitude[i][j] = noir_hw::abs_int64(x[i][j]);
        }
    }
    let mut squares = [[0; COLS]; N];
    for i in 0..N {
        for j in 0..COLS {
            squares[i][j] = mul(magnitude[i][j], magnitude[i][j]);
        }
    }
    let mut sums = [0; N];
    for i in 0..N {
        for j in 0..COLS {
            sums[i] += squares[i][j];
        }
    }
    let mut result = [0; N];
    for i in 0..N {
        sums[i].assert_max_bit_size::<64>();
        result[i] = sqrt(sums[i]);
    }
    result
}
