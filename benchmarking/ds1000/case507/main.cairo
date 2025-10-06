// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure faithfully preserved.

#[executable]
pub fn main() {
    // a = [1, 0, 3]
    let a = array![1_u32, 0_u32, 3_u32];

    // result =
    // [
    //   [0, 1, 0, 0],
    //   [1, 0, 0, 0],
    //   [0, 0, 0, 1]
    // ]
    let result = array![
        array![0_u32, 1_u32, 0_u32, 0_u32],
        array![1_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 1_u32],
    ];

    // Verify
    for i in 0..3_u32 {
        for j in 0..4_u32 {
            let expected: u32 = if *a.at(i) == j { 1_u32 } else { 0_u32 };
            assert!(*result.at(i).at(j) == expected);
        }
    }
}
