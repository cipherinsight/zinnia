// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a =
    // [[ 0,  1,  2,  3,  4],
    //  [ 5,  6,  7,  8,  9],
    //  [10, 11, 12, 13, 14],
    //  [15, 16, 17, 18, 19],
    //  [20, 21, 22, 23, 24]]
    let a = array![
        array![0_u32, 1_u32, 2_u32, 3_u32, 4_u32],
        array![5_u32, 6_u32, 7_u32, 8_u32, 9_u32],
        array![10_u32, 11_u32, 12_u32, 13_u32, 14_u32],
        array![15_u32, 16_u32, 17_u32, 18_u32, 19_u32],
        array![20_u32, 21_u32, 22_u32, 23_u32, 24_u32],
    ];

    // result = [4, 8, 12, 16, 20]
    let result = array![4_u32, 8_u32, 12_u32, 16_u32, 20_u32];

    // Step 1: flipped[i, j] = a[i, 4 - j] (horizontally flipped)
    let mut flipped = ArrayTrait::new();
    for i in 0..5_u32 {
        let mut row = ArrayTrait::new();
        for j in 0..5_u32 {
            let val = *a.at(i).at(4_u32 - j);
            row.append(val);
        }
        flipped.append(row);
    }

    // Step 2: diag_vals[k] = flipped[k, k]
    let mut diag_vals = ArrayTrait::new();
    for k in 0..5_u32 {
        diag_vals.append(*flipped.at(k).at(k));
    }

    // Step 3: Verify result == diag_vals
    for i in 0..5_u32 {
        assert!(*result.at(i) == *diag_vals.at(i));
    }
}
