// DS-1000 case 402 — Cairo baseline.
// Kernel: result = np.ravel_multi_index(index, dims=dims, order='F').
// 'F' (column-major): linear = sum_k index[k] * prod_{j<k} dims[j].
// Sample: dims=[8,9,8,8,8], index=[2,3,4,5,6] → 30842.
// `a` is unused by the math; we materialise it in Q-scale to mirror Zinnia's
// 36864 ReadFloatIR ops.

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
        // a[k] = k * 0.001 in Q-scale = k * Q / 1000.
        let kv: u128 = k.into();
        a.append((kv * Q) / 1000);
    };

    let mut index: Array<u128> = ArrayTrait::new();
    index.append(2);
    index.append(3);
    index.append(4);
    index.append(5);
    index.append(6);
    let claimed: u128 = 30842;

    let mut result: u128 = 0;
    let mut stride: u128 = 1;
    for k in 0_u32..5 {
        let ik: u128 = *index.at(k.try_into().unwrap());
        let dk: u128 = *dims.at(k.try_into().unwrap());
        result = result + ik * stride;
        stride = stride * dk;
    };

    assert!(result == claimed);
    // Force `a` to be live.
    assert!(a.len() == total);
}
