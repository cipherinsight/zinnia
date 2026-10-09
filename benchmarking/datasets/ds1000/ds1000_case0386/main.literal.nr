// DS-1000 case 0386, literal translation of kernel.py (LITERAL_BASELINES.md).
// Build windows in rows of 199 patches to avoid a giant flat-array SSA update.
// The nested ABI is flattened row-major, exactly as reshape(-1, 2, 2).
// @outputs: float[78804]
fn main(a: [Field; 20000]) -> pub [[[Field; 4]; 199]; 99] {
    noir_hw::check_inputs(a);
    // np.lib.stride_tricks.sliding_window_view(a, window_shape=(2, 2))
    let mut windows = [[[0; 4]; 199]; 99];
    for i in 0..99 {
        let mut row = [[0; 4]; 199];
        for j in 0..199 {
            let mut patch = [0; 4];
            for r in 0..2 {
                for s in 0..2 {
                    patch[r * 2 + s] = a[(i + r) * 200 + j + s];
                }
            }
            row[j] = patch;
        }
        windows[i] = row;
    }
    // .reshape(-1, 2, 2): row-major wires are unchanged.
    let result = windows;
    result
}
