// DS-1000 case 361 — np.delete(a, [0, 2], axis=1) on Float[6, 6] -> Float[6, 4].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // a[i][j] = (i*10 + j) + 0.1
    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..6 {
        for j in 0_u32..6 {
            let int_part: u128 = (i * 10 + j).into();
            let q_v: u128 = int_part * Q + Q / 10;
            data.append(q_v);
        };
    };
    let kept: Array<u32> = array![1, 3, 4, 5];
    let mut claimed: Array<u128> = ArrayTrait::new();
    for i in 0_u32..6 {
        for jo in 0_u32..4 {
            let j: u32 = *kept.at(jo);
            let int_part: u128 = (i * 10 + j).into();
            let q_v: u128 = int_part * Q + Q / 10;
            claimed.append(q_v);
        };
    };

    let mut result: Array<u128> = ArrayTrait::new();
    for i in 0_u32..6 {
        for jo in 0_u32..4 {
            let j: u32 = *kept.at(jo);
            result.append(*data.at(i * 6 + j));
        };
    };

    for k in 0_u32..24 {
        assert!(*result.at(k) == *claimed.at(k));
    };
}
