// pythran_specialconvolve__specialconvolve (Pythran specialconvolve), Noir baseline.
// Kernel: on `a: Int[R, C]`, rowconvol sums each 3-row window ([R - 2, C]),
// colconvol sums each 3-column window of it and subtracts 9 times the
// window's centre ([R - 2, C - 2], the output).
// Integer sums, exact in the field; the row sums are shared by the three
// column windows that use them.
// @outputs: int[(R - 2) * (C - 2)]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global R: u32 = 16; // = a_0
global C: u32 = 16; // = a_1

fn main(a: [[Field; C]; R]) -> pub [[Field; C - 2]; R - 2] {
    // Input domain: every input element is an int64.
    for i in 0..R {
        check_inputs(a[i]);
    }
    let mut rowconvol: [[Field; C]; R - 2] = [[0; C]; R - 2];
    for i in 0..R - 2 {
        for j in 0..C {
            rowconvol[i][j] = a[i + 1][j] + a[i][j] + a[i + 2][j];
        }
    }
    let mut colconvol: [[Field; C - 2]; R - 2] = [[0; C - 2]; R - 2];
    for i in 0..R - 2 {
        for j in 0..C - 2 {
            colconvol[i][j] = rowconvol[i][j + 1] + rowconvol[i][j] + rowconvol[i][j + 2] - 9 * a[i + 1][j + 1];
        }
    }
    colconvol
}
