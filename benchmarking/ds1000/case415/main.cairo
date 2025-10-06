// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    let data = array![4_u32, 2_u32, 5_u32, 6_u32, 7_u32, 5_u32, 4_u32, 3_u32, 5_u32, 7_u32];

    // result = [5, 7, 5]
    let result = array![5_u32, 7_u32, 5_u32];

    // bin_size = 3
    let bin_size: u32 = 3_u32;

    // trimmed = data[:(10 // bin_size) * bin_size]  -> first 9 elements
    let trimmed = array![4_u32, 2_u32, 5_u32, 6_u32, 7_u32, 5_u32, 4_u32, 3_u32, 5_u32];

    // reshaped = trimmed.reshape((3, bin_size))
    let reshaped = array![
        array![4_u32, 2_u32, 5_u32],
        array![6_u32, 7_u32, 5_u32],
        array![4_u32, 3_u32, 5_u32],
    ];

    // bin_data_max = reshaped.max(axis=1)
    let mut bin_data_max = ArrayTrait::new();
    for i in 0..3_u32 {
        let mut row_max: u32 = *reshaped.at(i).at(0_u32);
        for j in 1_u32..bin_size {
            let v = *reshaped.at(i).at(j);
            if v > row_max {
                row_max = v;
            }
        }
        bin_data_max.append(row_max);
    }

    // expected = bin_data_max
    // assert result == expected
    for k in 0..3_u32 {
        assert!(*result.at(k) == *bin_data_max.at(k));
    }
}
