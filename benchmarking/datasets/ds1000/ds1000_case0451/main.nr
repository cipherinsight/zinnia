// DS-1000 case 0451: optimized baseline.
// Constant construction at the declared shape; flattened output has 4000 values.
// @outputs: const float[4000] = 0
fn main() {
    // arr = np.zeros((20,10,10,2))
    let _result: [Field; 4000] = [0; 4000];
}
