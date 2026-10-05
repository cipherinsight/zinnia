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
    // [[0, 0, 0, 3],
    //  [0, 0, 0, 0],
    //  [1, 0, 0, 0],
    //  [0, 0, 0, 0]]
    let result = array![
        array![0_u32, 0_u32, 0_u32, 3_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32],
        array![1_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32],
    ];

    // zero_rows = [1, 3]
    // zero_cols = [1, 2]
    let zero_rows = array![1_u32, 3_u32];
    let zero_cols = array![1_u32, 2_u32];

    // Step 1: Set rows in zero_rows to 0
    let mut modified = ArrayTrait::new();
    for i in 0..4_u32 {
        let mut row = ArrayTrait::new();
        let mut zero_row: bool = false;
        for k in 0..2_u32 {
            if i == *zero_rows.at(k) {
                zero_row = true;
            }
        }
        for j in 0..4_u32 {
            if zero_row {
                row.append(0_u32);
            } else {
                row.append(*a.at(i).at(j));
            }
        }
        modified.append(row);
    }

    // Step 2: Set columns in zero_cols to 0
    let mut expected = ArrayTrait::new();
    for i in 0..4_u32 {
        let mut row = ArrayTrait::new();
        for j in 0..4_u32 {
            let mut zero_col: bool = false;
            for k in 0..2_u32 {
                if j == *zero_cols.at(k) {
                    zero_col = true;
                }
            }
            if zero_col {
                row.append(0_u32);
            } else {
                row.append(*modified.at(i).at(j));
            }
        }
        expected.append(row);
    }

    // Step 3: Verify result == expected
    for i in 0..4_u32 {
        for j in 0..4_u32 {
            assert!(*result.at(i).at(j) == *expected.at(i).at(j));
        }
    }
}
