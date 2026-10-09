// DS-1000 case 0440: literal translation of kernel.py (LITERAL_BASELINES.md).
// Preserve enumerate order, full diagonal sqrt array, then slice addition to initialized zeros; off-diagonal inputs unread.
// @outputs: float[50]
fn main(Y: [Field; 250]) -> pub [Field; 50] {
    let mut X = [0; 50];
    for i in 0..10 {
        let mut mat_diag = [0; 5];
        for j in 0..5 { mat_diag[j] = noir_hw::checked(Y[i * 25 + j * 6]); }
        let mut diag = [0; 5];
        for j in 0..5 { diag[j] = noir_hw::sqrt(mat_diag[j]); }
        for j in 0..5 { X[j * 10 + i] += diag[j]; }
    }
    X
}
