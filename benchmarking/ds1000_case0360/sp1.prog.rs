// DS-1000 case 360 — SP1 guest program.
// Kernel: np.delete(a, 2, axis=0) on Float[4, 4] -> Float[3, 4]. Keep rows 0, 1, 3.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [[f64; 4]; 4] = [[0.0; 4]; 4];
    for i in 0..4 { for j in 0..4 { a[i][j] = sp1_zkvm::io::read::<f64>(); } }
    let mut claimed: [[f64; 4]; 3] = [[0.0; 4]; 3];
    for i in 0..3 { for j in 0..4 { claimed[i][j] = sp1_zkvm::io::read::<f64>(); } }

    let kept: [usize; 3] = [0, 1, 3];
    let mut result: [[f64; 4]; 3] = [[0.0; 4]; 3];
    for (io, &i) in kept.iter().enumerate() {
        for j in 0..4 {
            result[io][j] = a[i][j];
        }
    }

    for i in 0..3 { for j in 0..4 {
        assert!((result[i][j] - claimed[i][j]).abs() < 1e-3);
    } }
}
