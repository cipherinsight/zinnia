// DS-1000 case 412 — SP1 guest program.
// Kernel: result = x[x >= 0] on Float[10]. Mirrors the halo2 baseline shape:
// fixed 10-cell output buffer; negatives are replaced by 0 (so the buffer is
// `select(x >= 0, x, 0)` per cell).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut x: [f64; 10] = [0.0; 10];
    for k in 0..10 { x[k] = sp1_zkvm::io::read::<f64>(); }
    let mut claimed: [f64; 10] = [0.0; 10];
    for k in 0..10 { claimed[k] = sp1_zkvm::io::read::<f64>(); }

    let zero: f64 = 0.0;
    for k in 0..10 {
        let lt = x[k] < zero;
        let ge = !lt;
        let kept: f64 = if ge { x[k] } else { 0.0 };
        assert!((kept - claimed[k]).abs() < 0.001);
    }
}
