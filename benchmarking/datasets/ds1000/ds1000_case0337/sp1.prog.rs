// DS-1000 case 337 — SP1 guest.
// `np.diag(np.fliplr(a))` on Float[8, 10] → 8 anti-diagonal cells a[i][9-i].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(10);
        for _ in 0..10 {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        a.push(row);
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(8);
    for _ in 0..8 {
        claimed.push(sp1_zkvm::io::read::<f64>());
    }

    for i in 0..8usize {
        let v = a[i][9 - i];
        assert!((v - claimed[i]).abs() < 1e-9);
    }
}
