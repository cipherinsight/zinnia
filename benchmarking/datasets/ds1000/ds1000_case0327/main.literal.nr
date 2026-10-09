// DS-1000 case 0327: literal translation of kernel.py (LITERAL_BASELINES.md).
// Constant padding creates all 17 entries; seven constant outputs are omitted from instances.
// @outputs: float[10]; const float[7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fn main(A: [Field; 10]) -> pub [Field; 10] {
    noir_hw::check_inputs(A);
    let mut result = [0; 17];
    for i in 0..10 { result[i] = A[i]; }
    for i in 10..17 { result[i] = 0; }
    let mut out = [0; 10];
    for i in 0..10 { out[i] = result[i]; }
    out
}
