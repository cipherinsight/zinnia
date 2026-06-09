// DS-1000 case 403 — Cairo baseline.
// Kernel: result = np.ravel_multi_index(index, dims=dims, order='C').
// 'C' (row-major): linear = ((((i0*d1 + i1)*d2 + i2)*d3 + i3)*d4 + i4).
// Sample: dims=[8,9,8,8,8], index=[2,3,4,5,6] → 11054.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut dims: Array<u128> = ArrayTrait::new();
    dims.append(8);
    dims.append(9);
    dims.append(8);
    dims.append(8);
    dims.append(8);

    let total: u32 = 8 * 9 * 8 * 8 * 8;
    let mut a: Array<u128> = ArrayTrait::new();
    for k in 0_u32..total {
        let kv: u128 = k.into();
        a.append((kv * Q) / 1000);
    };

    let mut index: Array<u128> = ArrayTrait::new();
    index.append(2);
    index.append(3);
    index.append(4);
    index.append(5);
    index.append(6);
    let claimed: u128 = 11054;

    let mut result: u128 = 0;
    for k in 0_u32..5 {
        let ik: u128 = *index.at(k.try_into().unwrap());
        let dk: u128 = *dims.at(k.try_into().unwrap());
        result = result * dk + ik;
    };

    assert!(result == claimed);
    assert!(a.len() == total);
}
