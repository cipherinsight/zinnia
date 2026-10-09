// pythran_create_grid__create_grid: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize zeros and each slice assignment in order; omit constant output column from publics.
// @outputs: repeat[r < N * N](float[2] | const float[1] = [0])

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0

fn main(x: [Field; N]) -> pub [[Field; 2]; N * N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(x);
    let mut z = [[[0; 3]; N]; N];
    for i in 0..N { for j in 0..N { z[i][j][0] = x[i]; } }
    for i in 0..N { for j in 0..N { z[i][j][1] = x[j]; } }
    let mut fast_grid = [[0; 2]; N*N];
    for i in 0..N { for j in 0..N { fast_grid[i*N+j] = [z[i][j][0],z[i][j][1]]; } }
    fast_grid
}
