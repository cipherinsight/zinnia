// DS-1000 case 329 — SP1 guest. Elementwise a ** 3 on Float[8, 6].
// Idiomatic Rust: native f64::powi.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a = [[0.0f64; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            a[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    let mut result = [[0.0f64; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            result[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }

    for i in 0..8 {
        for j in 0..6 {
            let computed = a[i][j].powi(3);
            assert!((computed - result[i][j]).abs() < 1e-6);
        }
    }
}
