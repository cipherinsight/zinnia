// DS-1000 case 329 — Cairo baseline. Elementwise a ** 3 on Float[8, 6].
//
// Quantisation ×2^32 to match Zinnia. a[i][j] = 1.0 + 0.1*(i+j) in float
// → q = ((10 + (i+j)) * Q) / 10 in Q-scale. Pow=3 unrolls into two
// fixed-point multiplications, each rescaling by /Q to stay in Q-scale,
// mirroring FixedPointChip::qmul.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u128..8 {
        for j in 0_u128..6 {
            a.append(((10 + (i + j)) * Q) / 10);
        };
    };
    // Expected: ((1 + (i+j)/10)³) in Q-scale, i.e. (10 + (i+j))³ * Q / 1000.
    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u128..8 {
        for j in 0_u128..6 {
            let base: u128 = 10 + (i + j);
            expected.append((base * base * base * Q) / 1000);
        };
    };

    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let base = *a.at((i * 6 + j).try_into().unwrap());
            let t1 = (base * base) / Q;  // q(a²)
            let t2 = (t1 * base) / Q;    // q(a³)
            assert!(t2 == *expected.at((i * 6 + j).try_into().unwrap()));
        }
    };
}
