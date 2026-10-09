// DS-1000 case 0303: literal translation of kernel.py (LITERAL_BASELINES.md).
// Only slice elements are read and domain-checked; discarded cells are not read.
// Reshape/copy is a complete array of wire aliases, with no arithmetic.
// @outputs: float[20]
fn main(A: [Field; 23]) -> pub [Field; 20] {
    let ncol: u32 = 5;
    let col: u32 = (23 / ncol) * ncol;
    let mut sliced = [0; 20];
    for i in 0..col { sliced[i] = A[i]; }
    noir_hw::check_inputs(sliced);
    let mut b = [[0; 5]; 4];
    for i in 0..4 { for j in 0..5 { b[i][j] = sliced[i * ncol + j]; } }
    let mut out = [0; 20];
    for i in 0..4 { for j in 0..5 { out[i * 5 + j] = b[i][j]; } }
    out
}
