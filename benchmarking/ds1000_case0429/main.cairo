// DS-1000 case 429 — Cairo baseline.
// Kernel: (a.mean() - 2*a.std(), a.mean() + 2*a.std()) on Float[30].
// Same shape as 0428 but ±2σ. Quantisation ×2^32; Newton sqrt; SqrtFIR ×2.

fn compute_std(a: @Array<i128>, mean: i128, n_q: i128, q: i128, qu: u128) -> i128 {
    let mut sq_sum: i128 = 0;
    for k in 0_u32..30 {
        let v: i128 = *a.at(k.try_into().unwrap());
        let d: i128 = v - mean;
        let d2: i128 = (d * d) / q;
        sq_sum = sq_sum + d2;
    };
    let var_q: i128 = (sq_sum * q) / n_q;
    let target: u128 = (var_q.try_into().unwrap()) * qu;
    let mut g: u128 = if target > 0 { target } else { 1 };
    let mut prev: u128 = 0;
    while g != prev {
        prev = g;
        g = (g + target / g) / 2;
    };
    let g_i: i128 = g.try_into().unwrap();
    g_i
}

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    const QU: u128 = 4_294_967_296;

    let mut a: Array<i128> = ArrayTrait::new();
    for i in 1_i128..31 { a.append(i * Q); };

    let mut sum: i128 = 0;
    for k in 0_u32..30 {
        sum = sum + *a.at(k.try_into().unwrap());
    };
    let n_q: i128 = 30 * Q;
    let mean: i128 = (sum * Q) / n_q;

    let std1: i128 = compute_std(@a, mean, n_q, Q, QU);
    let std2: i128 = compute_std(@a, mean, n_q, Q, QU);

    let two: i128 = 2 * Q;
    let two_std1: i128 = (two * std1) / Q;
    let two_std2: i128 = (two * std2) / Q;

    let lo: i128 = mean - two_std1;
    let hi: i128 = mean + two_std2;

    let claimed_lo: i128 = -7_777_682_818;
    let claimed_hi: i128 = 140_921_668_994;
    let tol: i128 = (5 * Q) / 1000;

    assert!(lo + tol >= claimed_lo);
    assert!(lo <= claimed_lo + tol);
    assert!(hi + tol >= claimed_hi);
    assert!(hi <= claimed_hi + tol);
}
