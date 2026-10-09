// DS-1000 case 0448: optimized baseline.
// Kernel: np.vstack(([x_dists.T], [y_dists.T])).T on two 3x4 arrays.
// The stacked shape is 2x4x3; final transpose has shape 3x4x2.
// All transposes and stacking are copies of wires, with no arithmetic.
// @outputs: float[24]
fn main(x_dists: [Field; 12], y_dists: [Field; 12]) -> pub [Field; 24] {
    noir_hw::check_inputs(x_dists);
    noir_hw::check_inputs(y_dists);
    let mut result = [0; 24];
    for i in 0..12 { result[2 * i] = x_dists[i]; result[2 * i + 1] = y_dists[i]; }
    result
}
