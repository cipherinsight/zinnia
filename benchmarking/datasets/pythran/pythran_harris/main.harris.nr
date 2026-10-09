// pythran_harris__harris (Pythran harris), Noir baseline.
// Kernel: Harris response det - k * tr * tr of I: Float[M, N] with
// dx[i, j] = I[i + 1, j + 1] - I[i, j + 1],
// dy[i, j] = I[i + 1, j + 1] - I[i + 1, j] (both (M - 1) x (N - 1)),
// A = dx * dx, B = dy * dy, C = dx * dy, tr = A + B, det = A * B - C * C and
// k = 0.05.
// Every `*` is an elementwise product NumPy materializes, so each is rounded
// (k * tr * tr is (k * tr) * tr). I[0, 0] reaches no output (the slices drop
// it), so it is not checked.
// @outputs: float[(M - 1) * (N - 1)]

use noir_hw::{check_input, mul};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 4; // = I_0
global N: u32 = 6; // = I_1

global K_Q: Field = 214748365; // round(0.05 * 2^32)

fn main(I: [[Field; N]; M]) -> pub [[Field; N - 1]; M - 1] {
    // Input domain: every element but I[0, 0] is a Q32 code in the window.
    for i in 0..M {
        for j in 0..N {
            if (i != 0) | (j != 0) {
                check_input(I[i][j]);
            }
        }
    }
    let mut result: [[Field; N - 1]; M - 1] = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            let dx = I[i + 1][j + 1] - I[i][j + 1];
            let dy = I[i + 1][j + 1] - I[i + 1][j];
            let a = mul(dx, dx);
            let b = mul(dy, dy);
            let c = mul(dx, dy);
            let tr = a + b;
            let det = mul(a, b) - mul(c, c);
            result[i][j] = det - mul(mul(K_Q, tr), tr);
        }
    }
    result
}
