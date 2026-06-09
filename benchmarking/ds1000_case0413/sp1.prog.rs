// DS-1000 case 413 — SP1 guest program.
// Kernel: `x[x.imag != 0]` on a real Float[10]. Since x.imag is identically 0
// for real inputs, the result is empty. We read the 10 inputs (mirroring the
// 10× ReadFloatIR Zinnia emits) but produce no output and make no assertion.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut x: [f64; 10] = [0.0; 10];
    for k in 0..10 {
        x[k] = sp1_zkvm::io::read::<f64>();
    }
    // x.imag is identically 0 for real input, so the mask is all-false.
    let _result: Vec<f64> = Vec::new();
    // touch x to defeat dead-code elimination
    let mut s: f64 = 0.0;
    for k in 0..10 { s += x[k] * 0.0; }
    assert!(s == 0.0);
}
