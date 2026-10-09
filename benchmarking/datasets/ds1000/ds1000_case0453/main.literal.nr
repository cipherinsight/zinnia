// DS-1000 case 0453: literal translation of kernel.py (LITERAL_BASELINES.md).
// Full elementwise intermediate, then index-order row reduction, then broadcast division; integer division directly yields Q32.
// Match the optimized signed-32 operand guards before the squared norm.
// @outputs: float[20]
fn main(X: [Field; 20]) -> pub [Field; 20] {
    noir_hw::check_inputs(X);
    for i in 0..20 { (X[i] + 2147483648).assert_max_bit_size::<32>(); }
    let mut squares = [0; 20];
    for i in 0..20 { squares[i] = X[i] * X[i]; }
    let mut norms = [0; 5];
    for i in 0..5 {
        let mut acc: Field = 0;
        for j in 0..4 { acc += squares[i*4+j]; }
        norms[i] = acc;
    }
    for i in 0..5 { norms[i] = noir_hw::sqrt(noir_hw::from_int(norms[i])); }
    let mut result = [0; 20];
    for i in 0..5 { for j in 0..4 {
        result[i*4+j] = noir_hw::div(noir_hw::from_int(X[i*4+j]), norms[i]);
    } }
    result
}
