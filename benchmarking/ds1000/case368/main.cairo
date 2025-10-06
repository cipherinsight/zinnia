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

    // result = 1
    let result: u32 = 1_u32;

    // comparison = (a == a[0]) elementwise
    // np.all(comparison, axis=-1) -> for each row r, all columns equal to row 0
    // Then assert that equals (result == 1) for each row.
    for r in 0..3_u32 {
        let mut row_all: bool = true;
        for c in 0..5_u32 {
            let eq: bool = *a.at(r).at(c) == *a.at(0_u32).at(c);
            row_all = row_all && eq;
        }
        let expect_true: bool = result == 1_u32;
        assert!(row_all == expect_true);
    }
}
