// DS-1000 case 409 — SP1 guest program.
// Kernel: z = x + y for Float[8, 6] each. Reads two 8×6 matrices and the
// claimed sum, then asserts elementwise equality.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row = Vec::with_capacity(6);
        for _ in 0..6 { row.push(sp1_zkvm::io::read::<f64>()); }
        x.push(row);
    }
    let mut y: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row = Vec::with_capacity(6);
        for _ in 0..6 { row.push(sp1_zkvm::io::read::<f64>()); }
        y.push(row);
    }
    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row = Vec::with_capacity(6);
        for _ in 0..6 { row.push(sp1_zkvm::io::read::<f64>()); }
        claimed.push(row);
    }

    for i in 0..8 {
        for j in 0..6 {
            let z = x[i][j] + y[i][j];
            assert!((z - claimed[i][j]).abs() < 1e-6);
        }
    }
}
