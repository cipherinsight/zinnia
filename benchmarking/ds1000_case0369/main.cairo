// DS-1000 case 369 — np.isclose(a, a[:, 0].reshape(-1, 1), atol=0, rtol=1e-5).all().
// Per row i, compare each a[i][j] against a[i][0] (broadcast first column).
// Quantisation ×2^32 = Q.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let rtol_q: u128 = 42_949;  // floor(1e-5 * Q)

    let a: Array<i128> = array![5, 5, 5, 7, 8, 7, 9, 9, 9];
    let expected: bool = false;

    let mut rtol_abs_col0: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i != 3 {
        let b: i128 = *a.at(i * 3);
        let abs_b: u128 = if b < 0 { (-b).try_into().unwrap() } else { b.try_into().unwrap() };
        let b_q: u128 = abs_b * Q;
        let prod: u128 = (b_q * rtol_q) / Q;
        rtol_abs_col0.append(prod);
        i = i + 1;
    };

    let mut all_close: bool = true;
    let mut ii: u32 = 0;
    while ii != 3 {
        let mut jj: u32 = 0;
        while jj != 3 {
            let v: i128 = *a.at(ii * 3 + jj);
            let b: i128 = *a.at(ii * 3);
            let diff: i128 = v - b;
            let abs_d_int: u128 = if diff < 0 { (-diff).try_into().unwrap() } else { diff.try_into().unwrap() };
            let abs_d_q: u128 = abs_d_int * Q;
            let leq: bool = abs_d_q <= *rtol_abs_col0.at(ii);
            all_close = all_close && leq;
            jj = jj + 1;
        };
        ii = ii + 1;
    };

    assert!(all_close == expected);
}
