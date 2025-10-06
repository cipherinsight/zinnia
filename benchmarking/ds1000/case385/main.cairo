// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved:
// 1) reshape (4x4) -> (2,2,2,2)
// 2) transpose axes twice: (0,2,1,3) then (1,0,2,3)  [overall: (2,0,1,3)]
// 3) reshape -> (4,2,2)

#[executable]
pub fn main() {
    // a =
    // [[1,  5,  9, 13],
    //  [2,  6, 10, 14],
    //  [3,  7, 11, 15],
    //  [4,  8, 12, 16]]
    let a = array![
        array![1_u32, 5_u32, 9_u32, 13_u32],
        array![2_u32, 6_u32, 10_u32, 14_u32],
        array![3_u32, 7_u32, 11_u32, 15_u32],
        array![4_u32, 8_u32, 12_u32, 16_u32],
    ];

    // result =
    // [
    //   [[1, 5],  [2, 6]],
    //   [[3, 7],  [4, 8]],
    //   [[9, 13], [10, 14]],
    //   [[11, 15],[12, 16]],
    // ]
    let result = array![
        array![array![1_u32, 5_u32],  array![2_u32, 6_u32]],
        array![array![3_u32, 7_u32],  array![4_u32, 8_u32]],
        array![array![9_u32, 13_u32], array![10_u32, 14_u32]],
        array![array![11_u32, 15_u32],array![12_u32, 16_u32]],
    ];

    // Emulate:
    // reshaped -> indices (i0,i1,i2,i3) in {0,1}^4 map to a[r,c] with:
    //   r = i0*2 + i1, c = i2*2 + i3
    // After transposes -> axes order (2,0,1,3) with indices (k0,k1,k2,k3) = (i2,i0,i1,i3)
    // Final reshape (4,2,2) with p = k0*2 + k1, inner (k2,k3)
    // => computed[p][k2][k3] = a[k1*2 + k2][k0*2 + k3]
    let mut computed = ArrayTrait::new(); // shape (4,2,2)
    for k0 in 0..2_u32 {
        for k1 in 0..2_u32 {
            let mut block = ArrayTrait::new(); // (2,2)
            for k2 in 0..2_u32 {
                let mut row = ArrayTrait::new(); // (2)
                for k3 in 0..2_u32 {
                    let r: u32 = k1 * 2_u32 + k2;
                    let c: u32 = k0 * 2_u32 + k3;
                    row.append(*a.at(r).at(c));
                }
                block.append(row);
            }
            computed.append(block);
        }
    }

    // Assert result == computed
    for p in 0..4_u32 {
        for r in 0..2_u32 {
            for c in 0..2_u32 {
                assert!(*result.at(p).at(r).at(c) == *computed.at(p).at(r).at(c));
            }
        }
    }
}
