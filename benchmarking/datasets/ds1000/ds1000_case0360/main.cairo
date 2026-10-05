// DS-1000 case 360 — np.delete(a, 2, axis=0) on Float[4, 4] -> Float[3, 4].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..4 {
        for j in 0_u32..4 {
            let num: u128 = ((i + 1) * 10 + (j + 1)).into();
            data.append((num * Q) / 10);
        };
    };
    let kept: Array<u32> = array![0, 1, 3];
    let mut claimed: Array<u128> = ArrayTrait::new();
    for io in 0_u32..3 {
        let i: u32 = *kept.at(io);
        for j in 0_u32..4 {
            let num: u128 = ((i + 1) * 10 + (j + 1)).into();
            claimed.append((num * Q) / 10);
        };
    };

    let mut result: Array<u128> = ArrayTrait::new();
    for io in 0_u32..3 {
        let i: u32 = *kept.at(io);
        for j in 0_u32..4 {
            result.append(*data.at(i * 4 + j));
        };
    };

    for k in 0_u32..12 {
        assert!(*result.at(k) == *claimed.at(k));
    };
}
