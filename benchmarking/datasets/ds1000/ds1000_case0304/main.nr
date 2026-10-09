// DS-1000 case 0304, Noir baseline.
// Kernel: A[3:][::-1], reshaped to (4, 5).
// @outputs: float[20]

fn main(A: [Field; 23]) -> pub [Field; 20] {
    let mut result = [0; 20];
    for i in 0..20 { result[i] = noir_hw::checked(A[22 - i]); }
    result
}
