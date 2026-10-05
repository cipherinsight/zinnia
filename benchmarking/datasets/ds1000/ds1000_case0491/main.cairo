// DS-1000 case 491 — Cairo baseline.
// (~np.isclose(s1, s2)).sum() with default atol=1e-8, rtol=1e-5.
// Q-scaled (×2^32). atol_q = floor(1e-8 * Q) = 42; rtol_q = floor(1e-5 * Q) = 42949.
// abs is realised as a signed branch on (a - b) since Cairo's u128 doesn't have it.
// |s2| is just s2 (sample inputs are non-negative). qmul = (a*b)/Q.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let atol_q: u128 = 42;
    let rtol_q: u128 = 42949;

    // s1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let s1: Array<u128> = array![
        1 * Q, 2 * Q, 3 * Q, 4 * Q, 5 * Q, 6 * Q,
    ];
    // s2 = [1.0, 2.5, 3.0, 4.1, 5.0, 6.000001]
    let s2: Array<u128> = array![
        1 * Q,
        (5 * Q) / 2,
        3 * Q,
        (41 * Q) / 10,
        5 * Q,
        6 * Q + Q / 1000000,
    ];
    let claimed: u128 = 2;

    let mut count: u128 = 0;
    for k in 0_u32..6 {
        let a = *s1.at(k);
        let b = *s2.at(k);
        let diff: u128 = if a >= b { a - b } else { b - a };
        let abs_b: u128 = b; // non-negative inputs
        let bound: u128 = atol_q + (rtol_q * abs_b) / Q;
        let close: bool = diff <= bound;
        let not_close: u128 = if close { 0 } else { 1 };
        count = count + not_close;
    };

    assert!(count == claimed);
}
