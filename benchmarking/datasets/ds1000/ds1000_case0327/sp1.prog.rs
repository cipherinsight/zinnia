// DS-1000 case 327 — SP1 guest program.
// Kernel: np.pad(A, (0, length - A.shape[0]), 'constant') for Float[10] + scalar length.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let length: u32 = sp1_zkvm::io::read::<u32>();
    let n: u32 = 10;
    let mut a: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n { a.push(sp1_zkvm::io::read::<f64>()); }
    let mut out: Vec<f64> = Vec::with_capacity(length as usize);
    for i in 0..length as usize {
        if i < n as usize { out.push(a[i]); } else { out.push(0.0); }
    }
    for v in &out {
        let claim: f64 = sp1_zkvm::io::read::<f64>();
        assert!((v - claim).abs() < 1e-9);
    }
}
