// pythran_create_grid__create_grid (Pythran create_grid), Noir baseline.
// Kernel: on x: Float[N], the grid z[i, j] = (x[i], x[j], 0) reshaped to
// (N * N, 3). Data movement only (NPB_BASELINES.md section 5, rule 6): the
// inputs are checked to their domain and copied to the outputs; the third
// column is the constant 0, not public.
// @outputs: repeat[r < N * N](float[2] | const float[1] = [0])

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0

fn main(x: [Field; N]) -> pub [[Field; 2]; N * N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(x);
    let mut grid: [[Field; 2]; N * N] = [[0; 2]; N * N];
    for i in 0..N {
        for j in 0..N {
            grid[i * N + j] = [x[i], x[j]];
        }
    }
    grid
}
