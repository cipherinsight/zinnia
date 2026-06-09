// DS-1000 case 309 — np argmax over a Float[8, 6] matrix.
//
// kernel.py:
//     @zk_circuit
//     def kernel(a: Float[Array, "8 6"]):
//         result = a.argmax()
//         return result
//
// Quantisation ×2^32 (Q) to match Zinnia's `precision_bits = 32`. argmax is
// monotonic so the index is identical regardless of scale; we keep the scale
// to match Zinnia's representation throughout.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    // Sample matrix: a[i][j] = ((i+1)*10 + (j+1)) / 100 in Float space.
    // Override a[3][2] with the deliberate max 99.0.
    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let q_v: u128 = if i == 3 && j == 2 {
                99_u128 * Q
            } else {
                let num: u128 = ((i + 1) * 10 + (j + 1)).into();
                (num * Q) / 100
            };
            data.append(q_v);
        };
    };
    let expected: u128 = 20;  // 3*6 + 2

    let mut best_val: u128 = 0;
    let mut best_idx: u128 = 0;
    for i in 0..8_u128 {
        for j in 0..6_u128 {
            let idx: u32 = (i * 6 + j).try_into().unwrap();
            let v = *data.at(idx.try_into().unwrap());
            if v > best_val {
                best_val = v;
                best_idx = i * 6 + j;
            }
        }
    };

    assert!(best_idx == expected);
}
