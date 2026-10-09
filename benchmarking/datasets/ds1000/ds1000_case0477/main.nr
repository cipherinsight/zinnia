// DS-1000 case 0477, optimized Noir baseline.
// np.diag: expose the four diagonal input wires; other outputs are constants.
// @outputs: float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]
fn main(i: [Field; 4]) -> pub [Field; 4] {
    noir_hw::check_inputs(i);
    i
}
