// DS-1000 case 0334: literal translation of kernel.py (LITERAL_BASELINES.md).
// Form the stack; mean is a running sum in axis order, followed by shared float division by 3.
// @outputs: float[50]
fn main(a: [Field; 50], b: [Field; 50], c: [Field; 50]) -> pub [Field; 50] {
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    noir_hw::check_inputs(c);
    let arrays = [a, b, c];
    let mut result = [0; 50];
    for i in 0..50 {
        let mut sum = arrays[0][i];
        for j in 1..3 { sum += arrays[j][i]; }
        result[i] = noir_hw::div(sum, 3 * noir_hw::SCALE);
    }
    result
}
