// DS-1000 case 0449: literal translation of kernel.py (LITERAL_BASELINES.md).
// Kernel: np.vstack(([x_dists.T], [y_dists.T])).T on two 3x4 arrays.
// The stacked shape is 2x4x3; final transpose has shape 3x4x2.
// All transposes and stacking are copies of wires, with no arithmetic.
// @outputs: float[24]
fn main(x_dists: [Field; 12], y_dists: [Field; 12]) -> pub [Field; 24] {
    noir_hw::check_inputs(x_dists);
    noir_hw::check_inputs(y_dists);
    // [x_dists.T], [y_dists.T]: each shape 1x4x3.
    let mut xt = [0; 12];
    let mut yt = [0; 12];
    for col in 0..4 {
        for row in 0..3 { xt[col * 3 + row] = x_dists[row * 4 + col]; }
    }
    for col in 0..4 {
        for row in 0..3 { yt[col * 3 + row] = y_dists[row * 4 + col]; }
    }
    // vstack gives 2x4x3; .T reverses all axes to 3x4x2.
    let mut stacked = [0; 24];
    for i in 0..12 { stacked[i] = xt[i]; stacked[12 + i] = yt[i]; }
    let mut result = [0; 24];
    for row in 0..3 {
        for col in 0..4 {
            for axis in 0..2 {
                result[(row * 4 + col) * 2 + axis] = stacked[axis * 12 + col * 3 + row];
            }
        }
    }
    result
}
