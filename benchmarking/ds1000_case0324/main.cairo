// DS-1000 case 324 — Cairo baseline. cos(deg2rad(degree)) via 4-term Taylor.
// Quantisation ×2^32 to match Zinnia's `precision_bits = 32` default.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;                  // 2^32
    let pi_over_180_q: u128 = 74_961_321;           // floor((π/180) · Q)
    let degree: u128 = 60;
    let claimed_q: u128 = Q / 2;                    // cos(60°) = 0.5
    let tol_q: u128 = (5 * Q) / 1000;               // ±0.005 tolerance

    // rad_q = degree (int) · pi_over_180_q (Q-scaled): integer-by-Q yields Q-scale.
    let rad_q: u128 = degree * pi_over_180_q;

    let x2_q: u128 = (rad_q * rad_q) / Q;
    let x4_q: u128 = (x2_q * x2_q) / Q;
    let x6_q: u128 = (x4_q * x2_q) / Q;

    let term2_q: u128 = x2_q / 2;
    let term4_q: u128 = x4_q / 24;
    let term6_q: u128 = x6_q / 720;

    let tmp1_q: u128 = Q - term2_q;
    let tmp2_q: u128 = tmp1_q + term4_q;
    let cos_q: u128 = tmp2_q - term6_q;

    assert!(cos_q + tol_q >= claimed_q);
    assert!(cos_q <= claimed_q + tol_q);
}
