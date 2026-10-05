// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // x =
    // [[2, 2, 2],
    //  [2, 2, 2],
    //  [2, 2, 2]]
    let x = array![
        array![2_u32, 2_u32, 2_u32],
        array![2_u32, 2_u32, 2_u32],
        array![2_u32, 2_u32, 2_u32],
    ];

    // y =
    // [[3, 3, 3],
    //  [3, 3, 3],
    //  [3, 3, 1]]
    let y = array![
        array![3_u32, 3_u32, 3_u32],
        array![3_u32, 3_u32, 3_u32],
        array![3_u32, 3_u32, 1_u32],
    ];

    // z =
    // [[5, 5, 5],
    //  [5, 5, 5],
    //  [5, 5, 3]]
    let z = array![
        array![5_u32, 5_u32, 5_u32],
        array![5_u32, 5_u32, 5_u32],
        array![5_u32, 5_u32, 3_u32],
    ];

    // expected = x + y (element-wise)
    let mut expected = ArrayTrait::new();
    for i in 0..3_u32 {
        let mut row = ArrayTrait::new();
        for j in 0..3_u32 {
            row.append(*x.at(i).at(j) + *y.at(i).at(j));
        }
        expected.append(row);
    }

    // assert z == expected
    for i in 0..3_u32 {
        for j in 0..3_u32 {
            assert!(*z.at(i).at(j) == *expected.at(i).at(j));
        }
    }
}
