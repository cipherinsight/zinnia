// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a =
    // [[0, 1],
    //  [2, 1],
    //  [4, 8]]
    let a = array![
        array![0_u32, 1_u32],
        array![2_u32, 1_u32],
        array![4_u32, 8_u32],
    ];

    // mask =
    // [[1, 0],
    //  [0, 1],
    //  [1, 0]]
    let mask = array![
        array![1_u32, 0_u32],
        array![0_u32, 1_u32],
        array![1_u32, 0_u32],
    ];

    // For each row, find the minimum and mark 1 where the element equals the row min.
    for i in 0..3_u32 {
        // row_min = a[i].min()
        let v0: u32 = *a.at(i).at(0_u32);
        let v1: u32 = *a.at(i).at(1_u32);
        let mut row_min: u32 = v0;
        if v1 < row_min {
            row_min = v1;
        }

        for j in 0..2_u32 {
            let eq: bool = *a.at(i).at(j) == row_min;
            let mut expected_bit: u32 = 0_u32;
            if eq {
                expected_bit = 1_u32;
            } else {
                expected_bit = 0_u32;
            }

            let m = *mask.at(i).at(j);
            // Constrain mask to be boolean and equal expected_bit
            assert!(m == 0_u32 || m == 1_u32);
            assert!(m == expected_bit);
        }
    }
}
