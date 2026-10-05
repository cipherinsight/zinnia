// DS-1000 case 428 — Cairo baseline.
// Kernel: (a.mean() - 3*a.std(), a.mean() + 3*a.std()) on Float[30].
// Sample a = [1..30]: mean=15.5, std≈8.65544.
//
// Quantisation ×2^32 to match Zinnia. Fixed-point sqrt via Newton on
// `target = var_q * Q` (so sqrt_q^2 / Q ≈ var_q). Zinnia emits SqrtFIR
// twice (the tuple re-evaluates a.std()); mirrored here. lo is negative
// so we use i128 with ±tol comparisons.

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

    let three: i128 = 3 * Q;
    let three_std1: i128 = (three * std1) / Q;
    let three_std2: i128 = (three * std2) / Q;

    let lo: i128 = mean - three_std1;
    let hi: i128 = mean + three_std2;

    let claimed_lo: i128 = -44_952_520_771;
    let claimed_hi: i128 = 178_096_506_947;
    let tol: i128 = (5 * Q) / 1000;

    assert!(lo + tol >= claimed_lo);
    assert!(lo <= claimed_lo + tol);
    assert!(hi + tol >= claimed_hi);
    assert!(hi <= claimed_hi + tol);
}
