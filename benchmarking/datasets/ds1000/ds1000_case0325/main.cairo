// DS-1000 case 325 — Cairo baseline.
// Kernel: `int(sin(n) > sin(deg2rad(n)))`. sin via 4-term Taylor in Q-scale.
// Sample n=1: sin(1 rad)≈0.841, sin(1°)≈0.0175 → result=1.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let pi_over_180_q: i128 = 74_961_321;  // floor((π/180) · Q)
    let number: i128 = 1;
    let claimed: u128 = 1;

    // n_q = number * Q (cast int to Q-scale).
    let n_q: i128 = number * Q;
    // rad = n_q * pi_over_180_q / Q.
    let rad_q: i128 = (n_q * pi_over_180_q) / Q;

    // sin via 4-term Taylor: x - x^3/6 + x^5/120 - x^7/5040.
    let sin_n: i128 = sin_taylor(n_q);
    let sin_deg: i128 = sin_taylor(rad_q);

    let result: u128 = if sin_n > sin_deg { 1 } else { 0 };
    assert!(result == claimed);
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
