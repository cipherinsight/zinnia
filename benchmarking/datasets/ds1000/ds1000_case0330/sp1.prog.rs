// DS-1000 case 330 — SP1 guest.
// `a ** power` on Float[8, 6] with integer scalar `power`.
// Repeated multiplication: result[i][j] = a[i][j]^power.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        a.push(row);
    }
    let power: u64 = sp1_zkvm::io::read::<u64>();

    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        claimed.push(row);
    }

    for i in 0..8usize {
        for j in 0..6usize {
            let mut acc: f64 = 1.0;
            for _ in 0..power {
                acc *= a[i][j];
            }
            assert!((acc - claimed[i][j]).abs() < 1e-9);
        }
    }
}
