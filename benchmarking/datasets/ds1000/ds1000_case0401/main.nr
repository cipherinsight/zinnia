// DS-1000 case 0401: optimized baseline.
// Constant construction at the declared shape; flattened output has 0 values.
// @outputs: const float[0] = 0
fn main() {
    // result = np.array([[], [], []])
    let _result: [Field; 0] = [0; 0];
}
