// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a =
    // [
    //   [1, 2, 3, 4, 5],
    //   [1, 2, 3, 4, 5],
    //   [1, 2, 3, 4, 5]
    // ]
    let a = array![
        array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32],
        array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32],
        array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32],
    ];

    // result = True (1)
    let result: u32 = 1_u32;

    // Meaning: check if all rows are identical.
    // comparison = a == a[0]
    // computed = np.all(comparison)
    let mut all_equal: bool = true;
    for i in 0..3_u32 {
        for j in 0..5_u32 {
            let eq: bool = *a.at(i).at(j) == *a.at(0_u32).at(j);
            all_equal = all_equal && eq;
        }
    }

    let expected_bool: bool = result == 1_u32;
    assert!(all_equal == expected_bool);
}
