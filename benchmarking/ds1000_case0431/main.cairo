// DS-1000 case 431 — Cairo baseline.
// Kernel: ~np.logical_and(a > mean-2σ, a < mean+2σ) on Integer[8].
// Sample: a = [1, 2, 3, 4, 5, 6, 7, 100]; mean=16, std≈31.8;
// lo≈-47.6, hi≈79.6. Only a[7]=100 is an outlier → mask = [0,…,0,1].
//
// Quantisation ×2^32. Integer inputs are lifted to Q-scale (FloatCast).
// Newton sqrt on var_q * Q. Comparison handles negative lo via i128.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    const QU: u128 = 4_294_967_296;

    let a_int: Array<i64> = array![1, 2, 3, 4, 5, 6, 7, 100];
    // FloatCast each int into Q-scale.
    let mut a: Array<i128> = ArrayTrait::new();
    for k in 0_u32..8 {
        let v: i64 = *a_int.at(k.try_into().unwrap());
        let v_i: i128 = v.into();
        a.append(v_i * Q);
    };

    let mut sum: i128 = 0;
    for k in 0_u32..8 {
        sum = sum + *a.at(k.try_into().unwrap());
    };
    let n_q: i128 = 8 * Q;
    let mean: i128 = (sum * Q) / n_q;

    let mut sq_sum: i128 = 0;
    for k in 0_u32..8 {
        let v: i128 = *a.at(k.try_into().unwrap());
        let d: i128 = v - mean;
        let d2: i128 = (d * d) / Q;
        sq_sum = sq_sum + d2;
    };
    let var_q: i128 = (sq_sum * Q) / n_q;
    let target: u128 = (var_q.try_into().unwrap()) * QU;
    let mut g: u128 = if target > 0 { target } else { 1 };
    let mut prev: u128 = 0;
    while g != prev {
        prev = g;
        g = (g + target / g) / 2;
    };
    let std_q: i128 = g.try_into().unwrap();

    let two: i128 = 2 * Q;
    let two_std: i128 = (two * std_q) / Q;
    let lo: i128 = mean - two_std;
    let hi: i128 = mean + two_std;

    let expected: Array<i128> = array![0, 0, 0, 0, 0, 0, 0, 1];
    for k in 0_u32..8 {
        let v: i128 = *a.at(k.try_into().unwrap());
        let gt_lo: bool = v > lo;
        let lt_hi: bool = v < hi;
        let inside: bool = gt_lo && lt_hi;
        let outside: i128 = if inside { 0 } else { 1 };
        assert!(outside == *expected.at(k.try_into().unwrap()));
    };
}
