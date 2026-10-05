// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a =
    // [
    //   [1, 1, 1],
    //   [2, 2, 2],
    //   [3, 3, 3],
    //   [4, 4, 4],
    //   [5, 5, 5]
    // ]
    let a = array![
        array![1_u32, 1_u32, 1_u32],
        array![2_u32, 2_u32, 2_u32],
        array![3_u32, 3_u32, 3_u32],
        array![4_u32, 4_u32, 4_u32],
        array![5_u32, 5_u32, 5_u32],
    ];

    // result = True (1)
    let result: u32 = 1_u32;

    // Meaning: verify that all columns are equal elementwise per row.
    // comparison = a == a[:,0].reshape((5,1))
    // computed = np.all(comparison)
    let mut all_equal: bool = true;
    for i in 0..5_u32 {
        let ref_val: u32 = *a.at(i).at(0_u32);
        for j in 0..3_u32 {
            let eq: bool = *a.at(i).at(j) == ref_val;
            all_equal = all_equal && eq;
        }
    }

    let expected_bool: bool = result == 1_u32;
    assert!(all_equal == expected_bool);
}
