// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure (trim → reshape → transpose → reshape) are faithfully preserved.

#[executable]
pub fn main() {
    // a: 4x5
    // [[ 1,  5,  9, 13, 17],
    //  [ 2,  6, 10, 14, 18],
    //  [ 3,  7, 11, 15, 19],
    //  [ 4,  8, 12, 16, 20]]
    let a = array![
        array![1_u32, 5_u32, 9_u32, 13_u32, 17_u32],
        array![2_u32, 6_u32, 10_u32, 14_u32, 18_u32],
        array![3_u32, 7_u32, 11_u32, 15_u32, 19_u32],
        array![4_u32, 8_u32, 12_u32, 16_u32, 20_u32],
    ];

    // expected result: 4 blocks of 2x2
    let result = array![
        array![array![1_u32, 5_u32], array![2_u32, 6_u32]],
        array![array![9_u32, 13_u32], array![10_u32, 14_u32]],
        array![array![3_u32, 7_u32], array![4_u32, 8_u32]],
        array![array![11_u32, 15_u32], array![12_u32, 16_u32]],
    ];

    // Patch size
    let patch: u32 = 2_u32;

    // 1) Trim to multiples of patch size
    let rows: u32 = 4_u32; // (a.shape[0] // 2) * 2
    let cols: u32 = 4_u32; // (a.shape[1] // 2) * 2
    // x = a[:rows, :cols] -> just ignore last column

    // 2) Blockify -> (rows/2, 2, cols/2, 2) == (2, 2, 2, 2)
    // blk[rblk][rinner][cblk][cinner] where:
    //   row = rblk*2 + rinner, col = cblk*2 + cinner
    let mut blk = ArrayTrait::new(); // [2][2][2][2]
    for rblk in 0..(rows / patch) {
        let mut lvl1 = ArrayTrait::new();
        for rinner in 0..patch {
            let mut lvl2 = ArrayTrait::new();
            for cblk in 0..(cols / patch) {
                let mut lvl3 = ArrayTrait::new();
                for cinner in 0..patch {
                    let row = rblk * patch + rinner;
                    let col = cblk * patch + cinner;
                    lvl3.append(*a.at(row).at(col));
                }
                lvl2.append(lvl3);
            }
            lvl1.append(lvl2);
        }
        blk.append(lvl1);
    }

    // 3) perm = blk.transpose((0, 2, 1, 3))
    // perm[rblk][cblk][rinner][cinner] = blk[rblk][rinner][cblk][cinner]
    let mut perm = ArrayTrait::new(); // [2][2][2][2]
    for rblk in 0..(rows / patch) {
        let mut lvl1 = ArrayTrait::new();
        for cblk in 0..(cols / patch) {
            let mut lvl2 = ArrayTrait::new();
            for rinner in 0..patch {
                let mut lvl3 = ArrayTrait::new();
                for cinner in 0..patch {
                    lvl3.append(*blk.at(rblk).at(rinner).at(cblk).at(cinner));
                }
                lvl2.append(lvl3);
            }
            lvl1.append(lvl2);
        }
        perm.append(lvl1);
    }

    // 4) Flatten blocks -> (num_blocks, 2, 2) == (4, 2, 2)
    // computed[p] with p = rblk*(cols/patch) + cblk
    let mut computed = ArrayTrait::new(); // [4][2][2]
    for rblk in 0..(rows / patch) {
        for cblk in 0..(cols / patch) {
            let mut block2 = ArrayTrait::new();
            for rinner in 0..patch {
                let mut block1 = ArrayTrait::new();
                for cinner in 0..patch {
                    block1.append(*perm.at(rblk).at(cblk).at(rinner).at(cinner));
                }
                block2.append(block1);
            }
            computed.append(block2);
        }
    }

    // Compare with provided result
    for i in 0..4_u32 {
        for r in 0..2_u32 {
            for c in 0..2_u32 {
                assert!(*result.at(i).at(r).at(c) == *computed.at(i).at(r).at(c));
            }
        }
    }
}
