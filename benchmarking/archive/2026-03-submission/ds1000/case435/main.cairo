// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure faithfully preserved.

#[executable]
pub fn main() {
    // a =
    // [[0, 3, 1, 3],
    //  [3, 0, 0, 0],
    //  [1, 0, 0, 0],
    //  [3, 0, 0, 0]]
    let a = array![
        array![0_u32, 3_u32, 1_u32, 3_u32],
        array![3_u32, 0_u32, 0_u32, 0_u32],
        array![1_u32, 0_u32, 0_u32, 0_u32],
        array![3_u32, 0_u32, 0_u32, 0_u32],
    ];

    // result =
    // [[0, 3, 1, 3],
    //  [0, 0, 0, 0],
    //  [0, 0, 0, 0],
    //  [0, 0, 0, 0]]
    let result = array![
        array![0_u32, 3_u32, 1_u32, 3_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32],
    ];

    // Zero out the 2nd row (index 1)
    let mut modified = ArrayTrait::new();
    for i in 0..4_u32 {
        let mut row = ArrayTrait::new();
        if i == 1_u32 {
            for _j in 0..4_u32 {
                row.append(0_u32);
            }
        } else {
            for j in 0..4_u32 {
                row.append(*a.at(i).at(j));
            }
        }
        modified.append(row);
    }

    // Then zero out the 1st column (index 0)
    let mut expected = ArrayTrait::new();
    for i in 0..4_u32 {
        let mut row = ArrayTrait::new();
        for j in 0..4_u32 {
            if j == 0_u32 {
                row.append(0_u32);
            } else {
                row.append(*modified.at(i).at(j));
            }
        }
        expected.append(row);
    }

    // Verify: result == expected
    for i in 0..4_u32 {
        for j in 0..4_u32 {
            assert!(*result.at(i).at(j) == *expected.at(i).at(j));
        }
    }
}
