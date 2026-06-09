// DS-1000 case 0496 — SP1 guest program.
// Kernel: np.pad(a, ((0, shape[0]-42), (0, shape[1]-13)), 'constant') with zeros.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const R: usize = 42;
    const C: usize = 10;
    let new_rows: u32 = sp1_zkvm::io::read::<u32>();
    let new_cols: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<f64> = Vec::with_capacity(R * C);
    for _ in 0..(R * C) { a.push(sp1_zkvm::io::read::<f64>()); }
    for i in 0..new_rows as usize {
        for j in 0..new_cols as usize {
            let v: f64 = if i < R && j < C { a[i * C + j] } else { 0.0 };
            let claim: f64 = sp1_zkvm::io::read::<f64>();
            assert!((v - claim).abs() < 1e-9);
        }
    }
}
