// DS-1000 case 339 — SP1 guest.
// `np.vstack((np.diag(b), np.diag(np.fliplr(b))))` where `b = a[:8, :8]`
// from Float[8, 10]. Equivalent to gathering a[i][i] and a[i][7-i] from the
// original 8×10 matrix.

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
    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut row: Vec<f64> = Vec::with_capacity(8);
        for _ in 0..8 {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        claimed.push(row);
    }

    for i in 0..8usize {
        let d = a[i][i];
        assert!((d - claimed[0][i]).abs() < 1e-9);
    }
    for i in 0..8usize {
        let f = a[i][7 - i];
        assert!((f - claimed[1][i]).abs() < 1e-9);
    }
}
