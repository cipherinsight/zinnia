// pythran_specialconvolve__specialconvolve: literal translation of kernel.py (LITERAL_BASELINES.md).
// Every elementwise intermediate is computed in full in expression order; integer arithmetic is exact.
// @outputs: int[(R - 2) * (C - 2)]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global R: u32 = 16; // = a_0
global C: u32 = 16; // = a_1

fn main(a: [[Field; C]; R]) -> pub [[Field; C - 2]; R - 2] {
    for i in 0..R {
        check_inputs(a[i]);
    }
    let mut row = [[0; C]; R - 2];
    for i in 0..R - 2 {
        for j in 0..C {
            row[i][j] = a[i + 1][j] + a[i][j];
        }
    }
    for i in 0..R - 2 {
        for j in 0..C {
            row[i][j] += a[i + 2][j];
        }
    }
    let mut col = [[0; C - 2]; R - 2];
    for i in 0..R - 2 {
        for j in 0..C - 2 {
            col[i][j] = row[i][j + 1] + row[i][j];
        }
    }
    for i in 0..R - 2 {
        for j in 0..C - 2 {
            col[i][j] += row[i][j + 2];
        }
    }
    let mut scaled = [[0; C - 2]; R - 2];
    for i in 0..R - 2 {
        for j in 0..C - 2 {
            scaled[i][j] = 9 * a[i + 1][j + 1];
        }
    }
    for i in 0..R - 2 {
        for j in 0..C - 2 {
            col[i][j] -= scaled[i][j];
        }
    }
    col
}
