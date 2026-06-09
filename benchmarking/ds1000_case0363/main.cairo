// DS-1000 case 363 — np.insert(a, pos, element) on Float[100] -> Float[101].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // a[i] = i * 0.1
    let mut a_arr: Array<u128> = ArrayTrait::new();
    for i in 0_u32..100 {
        let i_u: u128 = i.into();
        a_arr.append((i_u * Q) / 10);
    };
    let pos: u32 = 7;
    let element: u128 = 999;
    // element lifted to fp: element * Q
    let elem_q: u128 = element * Q;

    // Expected
    let mut claimed: Array<u128> = ArrayTrait::new();
    for i in 0_u32..101 {
        if i < pos {
            claimed.append(*a_arr.at(i));
        } else if i == pos {
            claimed.append(elem_q);
        } else {
            claimed.append(*a_arr.at(i - 1));
        }
    };

    let mut result: Array<u128> = ArrayTrait::new();
    for i in 0_u32..101 {
        if i < pos {
            result.append(*a_arr.at(i));
        } else if i == pos {
            result.append(elem_q);
        } else {
            result.append(*a_arr.at(i - 1));
        }
    };

    for k in 0_u32..101 {
        assert!(*result.at(k) == *claimed.at(k));
    };
}
