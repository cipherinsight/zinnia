// DS-1000 case 361 — SP1 guest program.
// Kernel: np.delete(a, [0, 2], axis=1) on Float[6, 6] -> Float[6, 4]. Keep cols 1, 3, 4, 5.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [[f64; 6]; 6] = [[0.0; 6]; 6];
    for i in 0..6 { for j in 0..6 { a[i][j] = sp1_zkvm::io::read::<f64>(); } }
    let mut claimed: [[f64; 4]; 6] = [[0.0; 4]; 6];
    for i in 0..6 { for j in 0..4 { claimed[i][j] = sp1_zkvm::io::read::<f64>(); } }

    let kept: [usize; 4] = [1, 3, 4, 5];
    let mut result: [[f64; 4]; 6] = [[0.0; 4]; 6];
    for i in 0..6 {
        for (jo, &j) in kept.iter().enumerate() {
            result[i][jo] = a[i][j];
        }
    }
    for i in 0..6 { for j in 0..4 {
        assert!((result[i][j] - claimed[i][j]).abs() < 1e-3);
    } }
}
