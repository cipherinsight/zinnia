// DS-1000 case 0439: literal translation of kernel.py (LITERAL_BASELINES.md).
// Transpose and newaxis are wire views; every broadcast product is computed, including both symmetric halves.
// @outputs: float[2000]
fn main(X: [Field; 200]) -> pub [Field; 2000] {
    noir_hw::check_inputs(X);
    let mut result = [0; 2000];
    for feature in 0..20 {
        for row in 0..10 {
            for col in 0..10 {
                result[feature * 100 + row * 10 + col] = noir_hw::mul(X[row * 20 + feature], X[col * 20 + feature]);
            }
        }
    }
    result
}
