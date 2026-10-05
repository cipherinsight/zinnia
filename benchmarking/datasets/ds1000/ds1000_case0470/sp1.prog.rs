// DS-1000 case 470 — SP1 guest program.
// kernel.py:
//     n = len(a); s = np.sum(a)
//     result = np.real(s)/n + 1j*np.imag(s)/n
// For Float[3] (real) input, np.imag(s) is 0, so the imaginary part is 0.
// We compute the real and imaginary parts separately and assert both against
// claimed values (tolerance ε = 0.001).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let a0: f64 = sp1_zkvm::io::read::<f64>();
    let a1: f64 = sp1_zkvm::io::read::<f64>();
    let a2: f64 = sp1_zkvm::io::read::<f64>();
    let claim_re: f64 = sp1_zkvm::io::read::<f64>();
    let claim_im: f64 = sp1_zkvm::io::read::<f64>();

    let n: f64 = 3.0;
    let s_re: f64 = a0 + a1 + a2;
    let s_im: f64 = 0.0;

    let res_re: f64 = s_re / n;
    let res_im: f64 = s_im / n;

    assert!((res_re - claim_re).abs() <= 0.001);
    assert!((res_im - claim_im).abs() <= 0.001);
}
