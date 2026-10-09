// DS-1000 case 0303, Noir baseline.
// Kernel: A[:20], reshaped to (4, 5).
// @outputs: float[20]

fn main(A: [Field; 23]) -> pub [Field; 20] {
    let mut result = [0; 20];
    for i in 0..20 { result[i] = noir_hw::checked(A[i]); }
    result
}
