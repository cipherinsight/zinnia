// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure (row slice [low, high)) are faithfully preserved.

#[executable]
pub fn main() {
    // a: NDArray[int, 3, 8]
    let a = array![
        array![0_u32, 1_u32, 2_u32, 3_u32, 5_u32, 6_u32, 7_u32, 8_u32],
        array![4_u32, 5_u32, 6_u32, 7_u32, 5_u32, 3_u32, 2_u32, 5_u32],
        array![8_u32, 9_u32, 10_u32, 11_u32, 4_u32, 5_u32, 3_u32, 5_u32],
    ];

    // result: NDArray[int, 2, 8]
    let result = array![
        array![0_u32, 1_u32, 2_u32, 3_u32, 5_u32, 6_u32, 7_u32, 8_u32],
        array![4_u32, 5_u32, 6_u32, 7_u32, 5_u32, 3_u32, 2_u32, 5_u32],
    ];

    // Slice bounds
    let low: u32 = 0_u32;
    let high: u32 = 2_u32; // exclusive
    let out_rows: u32 = high - low; // 2
    let ncols: u32 = 8_u32;

    // expected = a[low:high, :]
    let mut expected = ArrayTrait::new(); // [2][8]
    for i in low..high {
        let mut row = ArrayTrait::new();
        for j in 0_u32..ncols {
            row.append(*a.at(i).at(j));
        }
        expected.append(row);
    }

    // Assert result == expected
    for i in 0_u32..out_rows {
        for j in 0_u32..ncols {
            assert!(*result.at(i).at(j) == *expected.at(i).at(j));
        }
    }
}
