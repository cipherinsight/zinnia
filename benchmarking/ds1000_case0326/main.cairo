// DS-1000 case 326 — Cairo baseline.
// Kernel: `np.degrees(np.arcsin(value))`. arcsin is realised via the
// witness-and-verify pattern (mirrors halo2 qsqrt-style): witness y = asin(v),
// verify sin(y) ≈ v with 4-term Taylor, then scale by 180/π.
// Sample: value = 0.5 → degrees(asin(0.5)) = 30.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;                  // 2^32
    let one_q: i128 = Q;
    let one_neg_q: i128 = -Q;
    let pi_q: i128 = 13_493_037_705;                // floor(π · Q)
    let one_eighty_over_pi_q: i128 = 246_083_499_207;  // floor((180/π) · Q)
    let pi_half_q: i128 = pi_q / 2;

    let value_q: i128 = Q / 2;                      // 0.5
    let claimed_q: i128 = 30 * Q;                   // 30°
    let tol_q: i128 = (5 * Q) / 1000;               // ±0.005

    // Domain check: -1 ≤ value ≤ 1.
    assert!(value_q >= one_neg_q);
    assert!(value_q <= one_q);

    // Witnessed asin(0.5) = π/6.
    let asin_q: i128 = pi_q / 6;                    // π/6 in Q

    // Verify sin(asin_q) ≈ value_q via 4-term Taylor.
    let sin_y: i128 = sin_taylor(asin_q);
    let diff: i128 = sin_y - value_q;
    assert!(diff + tol_q >= 0);
    assert!(diff <= tol_q);

    // Range-check asin_q ∈ [-π/2, π/2].
    assert!(asin_q >= -pi_half_q);
    assert!(asin_q <= pi_half_q);

    // degrees = asin · 180/π.
    let deg_q: i128 = (asin_q * one_eighty_over_pi_q) / Q;
    assert!(deg_q + tol_q >= claimed_q);
    assert!(deg_q <= claimed_q + tol_q);
}

fn sin_taylor(x_q: i128) -> i128 {
    const Q: i128 = 4_294_967_296;
    let x2: i128 = (x_q * x_q) / Q;
    let x3: i128 = (x2 * x_q) / Q;
    let x5: i128 = (x3 * x2) / Q;
    let x7: i128 = (x5 * x2) / Q;
    let t1 = x_q;
    let t2 = x3 / 6;
    let t3 = x5 / 120;
    let t4 = x7 / 5040;
    (t1 - t2 + t3) - t4
}
