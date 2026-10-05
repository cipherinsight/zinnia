// DS-1000 case 359 — SP1 guest program.
// Kernel: np.delete(a, 2, axis=1) on Float[3, 3] -> Float[3, 2]. Keep cols 0, 1.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [[f64; 3]; 3] = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { a[i][j] = sp1_zkvm::io::read::<f64>(); } }
    let mut claimed: [[f64; 2]; 3] = [[0.0; 2]; 3];
    for i in 0..3 { for j in 0..2 { claimed[i][j] = sp1_zkvm::io::read::<f64>(); } }

    let kept: [usize; 2] = [0, 1];
    let mut result: [[f64; 2]; 3] = [[0.0; 2]; 3];
    for i in 0..3 {
        for (jo, &j) in kept.iter().enumerate() {
            result[i][jo] = a[i][j];
        }
    }

    for i in 0..3 { for j in 0..2 {
        assert!((result[i][j] - claimed[i][j]).abs() < 1e-3);
    } }
}
