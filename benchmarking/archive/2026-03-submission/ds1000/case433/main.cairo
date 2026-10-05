// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

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
    // [[0, 0, 0, 0],
    //  [0, 0, 0, 0],
    //  [0, 0, 0, 0],
    //  [0, 0, 0, 0]]
    let result = array![
        array![0_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32],
    ];

    // zero_rows = 0, zero_cols = 0
    let zero_rows: u32 = 0_u32;
    let zero_cols: u32 = 0_u32;

    // modified = a; modified[zero_rows, :] = 0
    let mut modified = ArrayTrait::new();
    for i in 0..4_u32 {
        let mut row = ArrayTrait::new();
        if i == zero_rows {
            // zero out the entire row
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

    // expected = modified; expected[:, zero_cols] = 0
    let mut expected = ArrayTrait::new();
    for i in 0..4_u32 {
        let mut row = ArrayTrait::new();
        for j in 0..4_u32 {
            if j == zero_cols {
                row.append(0_u32);
            } else {
                row.append(*modified.at(i).at(j));
            }
        }
        expected.append(row);
    }

    // assert result == expected
    for i in 0..4_u32 {
        for j in 0..4_u32 {
            assert!(*result.at(i).at(j) == *expected.at(i).at(j));
        }
    }
}
