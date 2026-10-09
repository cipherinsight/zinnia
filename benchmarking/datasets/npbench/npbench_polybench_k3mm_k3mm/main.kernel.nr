// npbench_polybench_k3mm_k3mm__kernel (NPBench polybench/k3mm), Noir baseline.
// Kernel: `A @ B @ C @ D` on A: Float[NI, NJ], B: Float[NJ, NK],
// C: Float[NK, NL], D: Float[NL, NM], evaluated left to right. Each element
// of each product is one reduction, rescaled once.
// @outputs: float[NI * NM]

use noir_hw::{check_inputs, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global NI: u32 = 13; // = A_0
global NJ: u32 = 15; // = A_1
global NK: u32 = 14; // = B_1
global NL: u32 = 16; // = C_1
global NM: u32 = 16; // = D_1

// x @ y for x: [R, K] and y: [K, C].
fn matmul<let R: u32, let K: u32, let C: u32>(x: [[Field; K]; R], y: [[Field; C]; K]) -> [[Field; C]; R] {
    let mut out: [[Field; C]; R] = [[0; C]; R];
    for i in 0..R {
        for j in 0..C {
            let mut acc: Field = 0;
            for k in 0..K {
                acc += x[i][k] * y[k][j];
            }
            out[i][j] = rescale(acc);
        }
    }
    out
}

fn main(
    A: [[Field; NJ]; NI],
    B: [[Field; NK]; NJ],
    C: [[Field; NL]; NK],
    D: [[Field; NM]; NL],
) -> pub [[Field; NM]; NI] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..NI {
        check_inputs(A[i]);
    }
    for i in 0..NJ {
        check_inputs(B[i]);
    }
    for i in 0..NK {
        check_inputs(C[i]);
    }
    for i in 0..NL {
        check_inputs(D[i]);
    }
    matmul(matmul(matmul(A, B), C), D)
}
