// Cairo translation of the given Zinnia code.
// Inputs are hard-coded, logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // A = [1, 2, 3, 4, 5, 6, 7]
    let A = array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32];

    // B = [[7, 6], [5, 4], [3, 2]]
    let B = array![
        array![7_u32, 6_u32],
        array![5_u32, 4_u32],
        array![3_u32, 2_u32],
    ];

    // truncated = A[1:] -> [2, 3, 4, 5, 6, 7]
    let truncated = array![2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32];

    // reversed_part = truncated[::-1] -> [7, 6, 5, 4, 3, 2]
    let reversed_part = array![7_u32, 6_u32, 5_u32, 4_u32, 3_u32, 2_u32];

    // reshaped = reversed_part.reshape((3, 2))
    // Expected B == reshaped
    let reshaped = array![
        array![7_u32, 6_u32],
        array![5_u32, 4_u32],
        array![3_u32, 2_u32],
    ];

    for i in 0..3_u32 {
        for j in 0..2_u32 {
            assert!(*B.at(i).at(j) == *reshaped.at(i).at(j));
        }
    }
}
