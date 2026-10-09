// DS-1000 case 0301: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reshape/copy is a complete array of wire aliases, with no arithmetic.
// @outputs: float[20]
fn main(A: [Field; 20]) -> pub [Field; 20] {
    noir_hw::check_inputs(A);
    let ncol: u32 = 5;
    let mut b = [[0; 5]; 4];
    for i in 0..4 { for j in 0..5 { b[i][j] = A[i * ncol + j]; } }
    let mut out = [0; 20];
    for i in 0..4 { for j in 0..5 { out[i * 5 + j] = b[i][j]; } }
    out
}
