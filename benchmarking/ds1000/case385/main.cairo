// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure (reshape → transpose → reshape) are faithfully preserved.

#[executable]
pub fn main() {
    // a: 4x4
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

    // result: 4x2x2
    let result = array![
        array![array![1_u32, 5_u32], array![2_u32, 6_u32]],
        array![array![3_u32, 7_u32], array![4_u32, 8_u32]],
        array![array![9_u32, 13_u32], array![10_u32, 14_u32]],
        array![array![11_u32, 15_u32], array![12_u32, 16_u32]],
    ];

    // ---- Step 1: reshape a (4,4) -> reshaped (2,2,2,2)
    // Mapping (row, col) = (i0*2 + i2, i1*2 + i3)
    let mut reshaped = ArrayTrait::new(); // [2][2][2][2]
    for i0 in 0..2_u32 {
        let mut blk0 = ArrayTrait::new();
        for i1 in 0..2_u32 {
            let mut blk1 = ArrayTrait::new();
            for i2 in 0..2_u32 {
                let mut blk2 = ArrayTrait::new();
                for i3 in 0..2_u32 {
                    let row = i0 * 2_u32 + i2;
                    let col = i1 * 2_u32 + i3;
                    blk2.append(*a.at(row).at(col));
                }
                blk1.append(blk2);
            }
            blk0.append(blk1);
        }
        reshaped.append(blk0);
    }

    // ---- Step 2a: transpose axes (0,1,2,3) -> (0,2,1,3)
    let mut t1 = ArrayTrait::new(); // [2][2][2][2]
    for a0 in 0..2_u32 {
        let mut x0 = ArrayTrait::new();
        for a2 in 0..2_u32 {
            let mut x1 = ArrayTrait::new();
            for a1 in 0..2_u32 {
                let mut x2 = ArrayTrait::new();
                for a3 in 0..2_u32 {
                    // t1[a0][a2][a1][a3] = reshaped[a0][a1][a2][a3]
                    x2.append(*reshaped.at(a0).at(a1).at(a2).at(a3));
                }
                x1.append(x2);
            }
            x0.append(x1);
        }
        t1.append(x0);
    }

    // ---- Step 2b: transpose axes (0,1,2,3) -> (1,0,2,3) on t1
    // Overall effect from original: (0,1,2,3) -> (2,0,1,3)
    // We implement the two-step sequence exactly.
    let mut t2 = ArrayTrait::new(); // [2][2][2][2]
    for b0 in 0..2_u32 {
        let mut y0 = ArrayTrait::new();
        for b1 in 0..2_u32 {
            let mut y1 = ArrayTrait::new();
            for b2 in 0..2_u32 {
                let mut y2 = ArrayTrait::new();
                for b3 in 0..2_u32 {
                    // t2[b0][b1][b2][b3] = t1[b1][b0][b2][b3]
                    y2.append(*t1.at(b1).at(b0).at(b2).at(b3));
                }
                y1.append(y2);
            }
            y0.append(y1);
        }
        t2.append(y0);
    }
    // Note: After both transposes, the logical axis order is (2,0,1,3).

    // ---- Step 3: reshape t2 (2,2,2,2) -> computed (4,2,2)
    // Combine first two axes into one: p = c*2 + a, where indices in t2 are [c][a][b][d].
    let mut computed = ArrayTrait::new(); // [4][2][2]
    for p in 0..4_u32 {
        let c = p / 2_u32;
        let a_idx = p % 2_u32;

        let mut row2 = ArrayTrait::new();
        for b in 0..2_u32 {
            let mut row1 = ArrayTrait::new();
            for d in 0..2_u32 {
                row1.append(*t2.at(c).at(a_idx).at(b).at(d));
            }
            row2.append(row1);
        }
        computed.append(row2);
    }

    // ---- Compare with provided result
    for i in 0..4_u32 {
        for r in 0..2_u32 {
            for s in 0..2_u32 {
                assert!(*result.at(i).at(r).at(s) == *computed.at(i).at(r).at(s));
            }
        }
    }
}
