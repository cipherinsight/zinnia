// DS-1000 case 336 — Cairo baseline. np.diag(np.fliplr(a)) on Float[8, 8].
//
// Driver: a[i][j] = 1.0 + 0.1·(i*8 + j). Quantisation ×2^32 to match Zinnia.
// Pure static gather; truncation in /10 is identical for input and expected,
// so equality holds exactly.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u128..8 {
        for j in 0_u128..8 {
            a.append(((10 + i * 8 + j) * Q) / 10);
        };
    };

    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u128..8 {
        expected.append(((10 + i * 8 + (7 - i)) * Q) / 10);
    };

    for i in 0_u32..8 {
        let j_flipped: u32 = 7 - i;
        let idx: u32 = i * 8 + j_flipped;
        let computed = *a.at(idx.try_into().unwrap());
        assert!(computed == *expected.at(i.try_into().unwrap()));
    };
}
