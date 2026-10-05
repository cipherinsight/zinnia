// DS-1000 case 500 — SP1 guest program.
// kernel.py: a.reshape(-1, 3) on Float[54] → Float[18, 3].
// Reshape is layout-only — output values are the input flat sequence.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 54;
    let mut a: [f64; N] = [0.0; N];
    for k in 0..N { a[k] = sp1_zkvm::io::read::<f64>(); }
    let mut claimed: [f64; N] = [0.0; N];
    for k in 0..N { claimed[k] = sp1_zkvm::io::read::<f64>(); }

    // Re-emit in row-major (i.e. unchanged order).
    for i in 0..18 {
        for j in 0..3 {
            let idx = i * 3 + j;
            let v = a[idx];
            assert!((v - claimed[idx]).abs() <= 0.001);
        }
    }
}
