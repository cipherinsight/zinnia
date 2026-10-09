// pythran_matmul__matrix_multiply: literal translation of kernel.py (LITERAL_BASELINES.md).
// zero preserves m rows and n columns; products round in the inner recurrence.
// @outputs: float[N * P]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = m0_0
global K: u32 = 16; // = m0_1
global P: u32 = 16; // = m1_1

fn zero() -> [[Field; N]; P] {
    let mut result = [[0; N]; P];
    for col in 0..P { for row in 0..N { result[col][row] = 0; } }
    result
}

fn main(m0: [[Field; K]; N], m1: [[Field; P]; K]) -> pub [[Field; N]; P] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(m0[i]);
    }
    for k in 0..K {
        check_inputs(m1[k]);
    }
    let mut result = zero();
    for i in 0..N {
        for j in 0..P {
            // r = 0; for k in range(K): r += m0[i][k] * m1[k][j]
            let mut r: Field = 0;
            for k in 0..K {
                r += mul(m0[i][k], m1[k][j]);
            }
            result[i][j] = r;
        }
    }
    result
}
