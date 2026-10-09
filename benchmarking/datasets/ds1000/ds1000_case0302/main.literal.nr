// DS-1000 case 0302: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reshape/copy is a complete array of wire aliases, with no arithmetic.
// @outputs: float[20]
fn main(A: [Field; 20]) -> pub [Field; 20] {
    noir_hw::check_inputs(A);
    let nrow: u32 = 5;
    let ncol: u32 = 20 / nrow;
    let mut b = [[0; 4]; 5];
    for i in 0..5 { for j in 0..4 { b[i][j] = A[i * ncol + j]; } }
    let mut out = [0; 20];
    for i in 0..5 { for j in 0..4 { out[i * ncol + j] = b[i][j]; } }
    out
}
