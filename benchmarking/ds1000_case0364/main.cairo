// DS-1000 case 364 — np.insert(a, pos, element, axis=0) on Float[100, 10] -> Float[101, 10].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // a[i][j] = i + j * 0.01
    let mut a_arr: Array<u128> = ArrayTrait::new();
    for i in 0_u32..100 {
        for j in 0_u32..10 {
            let i_u: u128 = i.into();
            let j_u: u128 = j.into();
            // i*Q + j*Q/100
            let q_v: u128 = i_u * Q + (j_u * Q) / 100;
            a_arr.append(q_v);
        };
    };
    let pos: u32 = 5;
    let mut element: Array<u128> = ArrayTrait::new();
    for j in 0_u32..10 {
        let j_u: u128 = j.into();
        element.append(999 * Q + j_u * Q);
    };

    // expected
    let mut claimed: Array<u128> = ArrayTrait::new();
    for i in 0_u32..101 {
        if i < pos {
            for j in 0_u32..10 { claimed.append(*a_arr.at(i * 10 + j)); };
        } else if i == pos {
            for j in 0_u32..10 { claimed.append(*element.at(j)); };
        } else {
            for j in 0_u32..10 { claimed.append(*a_arr.at((i - 1) * 10 + j)); };
        }
    };

    let mut result: Array<u128> = ArrayTrait::new();
    for i in 0_u32..101 {
        if i < pos {
            for j in 0_u32..10 { result.append(*a_arr.at(i * 10 + j)); };
        } else if i == pos {
            for j in 0_u32..10 { result.append(*element.at(j)); };
        } else {
            for j in 0_u32..10 { result.append(*a_arr.at((i - 1) * 10 + j)); };
        }
    };

    for k in 0_u32..1010 {
        assert!(*result.at(k) == *claimed.at(k));
    };
}
