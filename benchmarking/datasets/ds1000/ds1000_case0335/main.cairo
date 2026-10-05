// DS-1000 case 335 — Cairo baseline.
// Kernel: `np.max([a, b, c], axis=0)` over three Float[50] arrays.
// Quantisation ×2^32; per-cell max via two compare-and-select steps.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    let mut b: Array<u128> = ArrayTrait::new();
    let mut c: Array<u128> = ArrayTrait::new();
    let mut claimed: Array<u128> = ArrayTrait::new();

    for k in 0_u128..50 {
        // a[k] = 0.10 + 0.01*k = (10 + k) * Q / 100
        let a_q: u128 = ((10 + k) * Q) / 100;
        // b[k] = 0.20 + 0.015*k = (200 + 15*k) * Q / 1000
        let b_q: u128 = ((200 + 15 * k) * Q) / 1000;
        // c[k] = 0.05 + 0.02*k = (5 + 2*k) * Q / 100
        let c_q: u128 = ((5 + 2 * k) * Q) / 100;
        a.append(a_q);
        b.append(b_q);
        c.append(c_q);

        let ab = if a_q > b_q { a_q } else { b_q };
        let m = if ab > c_q { ab } else { c_q };
        claimed.append(m);
    };

    for k in 0_u32..50 {
        let a_q = *a.at(k);
        let b_q = *b.at(k);
        let c_q = *c.at(k);
        let ab = if a_q > b_q { a_q } else { b_q };
        let m = if ab > c_q { ab } else { c_q };
        assert!(m == *claimed.at(k));
    };
}
