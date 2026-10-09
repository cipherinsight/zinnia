// DS-1000 case 0400: literal translation of kernel.py (LITERAL_BASELINES.md).
// Constant construction at the declared shape; flattened output has 0 values.
// @outputs: const float[0] = 0
fn main() {
    // result = np.array([])
    let _result: [Field; 0] = [0; 0];
}
