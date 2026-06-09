// DS-1000 case 503 — Cairo baseline.
// kernel.py:
//     result = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//   a: Float[13,13,5], b: Integer[13,13]; output: Float[13,13].
//
// Quantisation ×2^32 to match Zinnia's `precision_bits = 32`. Pure gather, so
// Q-scaled equality is exact.
//
// Sample inputs:
//   a[i][j][k] = (i*65 + j*5 + k + 1) / 100
//   b[i][j]    = (i + j) % 5

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            for k in 0_u128..5 {
                let num: u128 = i * 65 + j * 5 + k + 1;
                a.append((num * Q) / 100);
            };
        };
    };

    let mut b: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            b.append((i + j) % 5);
        };
    };

    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let bij: u128 = (i + j) % 5;
            let num: u128 = i * 65 + j * 5 + bij + 1;
            expected.append((num * Q) / 100);
        };
    };

    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let bij: u128 = *b.at((i * 13 + j).try_into().unwrap());
            let a_idx: u32 = (i * 65 + j * 5 + bij).try_into().unwrap();
            let v: u128 = *a.at(a_idx);
            let e: u128 = *expected.at((i * 13 + j).try_into().unwrap());
            assert!(v == e);
        };
    };
}
