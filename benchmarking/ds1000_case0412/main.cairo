// DS-1000 case 412 — Cairo baseline.
// Kernel: x[x >= 0] on Float[10]. Same shape as the halo2 baseline (target.rs):
// 10-cell output buffer; negatives are replaced by 0.
// Sample: x = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0].
// Expected = [0.1, 0, 0.3, 0, 0.5, 0, 0.7, 0, 0.9, 0].
// Quantisation ×2^32; signed i128 (sign bit indicates negatives).

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;

    let mut x: Array<i128> = ArrayTrait::new();
    x.append(Q / 10);
    x.append(-(2 * Q) / 10);
    x.append((3 * Q) / 10);
    x.append(-(4 * Q) / 10);
    x.append((5 * Q) / 10);
    x.append(-(6 * Q) / 10);
    x.append((7 * Q) / 10);
    x.append(-(8 * Q) / 10);
    x.append((9 * Q) / 10);
    x.append(-Q);

    let mut expected: Array<i128> = ArrayTrait::new();
    expected.append(Q / 10);
    expected.append(0);
    expected.append((3 * Q) / 10);
    expected.append(0);
    expected.append((5 * Q) / 10);
    expected.append(0);
    expected.append((7 * Q) / 10);
    expected.append(0);
    expected.append((9 * Q) / 10);
    expected.append(0);

    for k in 0_u32..10 {
        let v: i128 = *x.at(k.try_into().unwrap());
        let lt: bool = v < 0;
        let ge: bool = !lt;
        let kept: i128 = if ge { v } else { 0 };
        let want: i128 = *expected.at(k.try_into().unwrap());
        assert!(kept == want);
    };
}
