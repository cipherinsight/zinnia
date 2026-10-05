// DS-1000 case 317 — SP1 guest program.
// Kernel: np.array(a) on Float[10, 20, 5] — identity copy of 1000 cells.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        a.push(sp1_zkvm::io::read::<f64>());
    }

    // Identity copy.
    let mut out: Vec<f64> = Vec::with_capacity(1000);
    for i in 0..1000 {
        out.push(a[i]);
    }

    for i in 0..1000 {
        let claim_v: f64 = sp1_zkvm::io::read::<f64>();
        assert!((out[i] - claim_v).abs() < 1e-12);
    }
}
