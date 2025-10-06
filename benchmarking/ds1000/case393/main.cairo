// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure (column slice [low, clamped_high)) are faithfully preserved.

#[executable]
pub fn main() {
    // a: NDArray[int, 3, 8]
    let a = array![
        array![0_u32, 1_u32, 2_u32, 3_u32, 5_u32, 6_u32, 7_u32, 8_u32],
        array![4_u32, 5_u32, 6_u32, 7_u32, 5_u32, 3_u32, 2_u32, 5_u32],
        array![8_u32, 9_u32, 10_u32, 11_u32, 4_u32, 5_u32, 3_u32, 5_u32],
    ];

    // result: NDArray[int, 3, 7]
    let result = array![
        array![1_u32, 2_u32, 3_u32, 5_u32, 6_u32, 7_u32, 8_u32],
        array![5_u32, 6_u32, 7_u32, 5_u32, 3_u32, 2_u32, 5_u32],
        array![9_u32, 10_u32, 11_u32, 4_u32, 5_u32, 3_u32, 5_u32],
    ];

    // Slice bounds
    let low: u32 = 1_u32;
    let clamped_high: u32 = 8_u32; // min(10, a.shape[1]) where a.shape[1] = 8
    let out_cols: u32 = clamped_high - low; // 7

    // expected = a[:, low:clamped_high]
    let mut expected = ArrayTrait::new(); // [3][7]
    for i in 0..3_u32 {
        let mut row = ArrayTrait::new();
        for j in low..clamped_high {
            row.append(*a.at(i).at(j));
        }
        expected.append(row);
    }

    // Assert result == expected
    for i in 0..3_u32 {
        for k in 0..out_cols {
            assert!(*result.at(i).at(k) == *expected.at(i).at(k));
        }
    }
}
