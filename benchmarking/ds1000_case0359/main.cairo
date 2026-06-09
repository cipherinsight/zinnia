// DS-1000 case 359 — np.delete(a, 2, axis=1) on Float[3, 3] -> Float[3, 2].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // a[i][j] = ((i+1)*10 + (j+1)) / 10
    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..3 {
            let num: u128 = ((i + 1) * 10 + (j + 1)).into();
            data.append((num * Q) / 10);
        };
    };
    // Expected: drop col 2 -> a[i][0], a[i][1].
    let mut claimed: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..2 {
            let num: u128 = ((i + 1) * 10 + (j + 1)).into();
            claimed.append((num * Q) / 10);
        };
    };

    let kept: Array<u32> = array![0, 1];
    let mut result: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for jo in 0_u32..2 {
            let j: u32 = *kept.at(jo);
            result.append(*data.at(i * 3 + j));
        };
    };

    for k in 0_u32..6 {
        assert!(*result.at(k) == *claimed.at(k));
    };
}
