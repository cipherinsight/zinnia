// DS-1000 case 450 — `a[:, second.reshape(-1,1), third]`.
// a: Float[7,8,9], second, third: Integer[2]. Result: Float[7,2,2].
// result[i][p][q] = a[i][second[p]][third[q]].
// Quantisation ×2^32 (Q). a[i][j][k] encodes i*100 + j*10 + k.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..7 {
        for j in 0_u32..8 {
            for k in 0_u32..9 {
                let v: u128 = (i * 100 + j * 10 + k).into();
                a.append(v * Q);
            };
        };
    };
    let second: Array<u32> = array![2_u32, 5_u32];
    let third: Array<u32> = array![3_u32, 7_u32];

    let mut result: Array<u128> = ArrayTrait::new();
    for i in 0_u32..7 {
        for p in 0_u32..2 {
            for q in 0_u32..2 {
                let j: u32 = *second.at(p);
                let k: u32 = *third.at(q);
                let idx: u32 = i * 72 + j * 9 + k;
                result.append(*a.at(idx));
            };
        };
    };

    // Claimed is the same map, computed directly.
    for i in 0_u32..7 {
        for p in 0_u32..2 {
            for q in 0_u32..2 {
                let j: u32 = *second.at(p);
                let k: u32 = *third.at(q);
                let expected: u128 = ((i * 100 + j * 10 + k).into()) * Q;
                let off: u32 = (i * 2 + p) * 2 + q;
                assert!(*result.at(off) == expected);
            };
        };
    };
}
