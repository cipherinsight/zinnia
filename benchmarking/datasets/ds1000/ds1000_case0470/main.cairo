// DS-1000 case 470 — Cairo baseline. Mean of Float[3] returned as (real, imag).
// imag part is 0 because input is real. Inputs Q-scaled by 2^32.
// Sample: a = [1.5, 2.5, 5.0] → sum = 9.0, mean = 3.0.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let tol_q: u128 = Q / 1000; // ±0.001

    let a0_q: u128 = (15 * Q) / 10;
    let a1_q: u128 = (25 * Q) / 10;
    let a2_q: u128 = 5 * Q;
    let claim_re_q: u128 = 3 * Q;
    let claim_im_q: u128 = 0;
    let n_q: u128 = 3 * Q;

    let s_re_q: u128 = a0_q + a1_q + a2_q;
    let s_im_q: u128 = 0;

    // Division: (a/b) in Q-scale = (a * Q) / b.
    let res_re_q: u128 = (s_re_q * Q) / n_q;
    let res_im_q: u128 = if n_q == 0 { 0 } else { (s_im_q * Q) / n_q };

    assert!(res_re_q + tol_q >= claim_re_q);
    assert!(res_re_q <= claim_re_q + tol_q);
    assert!(res_im_q + tol_q >= claim_im_q);
    assert!(res_im_q <= claim_im_q + tol_q);
}
