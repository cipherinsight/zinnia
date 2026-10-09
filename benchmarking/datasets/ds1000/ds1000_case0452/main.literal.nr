// DS-1000 case 0452: literal translation of kernel.py (LITERAL_BASELINES.md).
// Full elementwise intermediate, then index-order row reduction, then broadcast division; integer division directly yields Q32.
// @outputs: float[20]
fn main(X: [Field; 20]) -> pub [Field; 20] {
    noir_hw::check_inputs(X);
    let mut magnitudes = [0; 20];
    for i in 0..20 { magnitudes[i] = noir_hw::abs_int64(X[i]); }
    let mut norms = [0; 5];
    for i in 0..5 {
        let mut acc: Field = 0;
        for j in 0..4 { acc += magnitudes[i*4+j]; }
        norms[i] = acc;
    }
    let mut result = [0; 20];
    for i in 0..5 { for j in 0..4 {
        result[i*4+j] = noir_hw::div(X[i*4+j], norms[i]);
    } }
    result
}
