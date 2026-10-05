#[executable]
pub fn main() {
    let input = array![array![0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32], array![0_u32, 0_u32, 5_u32, 1_u32, 2_u32, 0_u32], array![0_u32, 1_u32, 8_u32, 0_u32, 1_u32, 0_u32], array![0_u32, 0_u32, 0_u32, 7_u32, 1_u32, 0_u32], array![0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32]];
    let result = array![array![0_u32, 5_u32, 1_u32, 2_u32], array![1_u32, 8_u32, 0_u32, 1_u32], array![0_u32, 0_u32, 7_u32, 1_u32]];

    let mut zero_rows = ArrayTrait::new();
    for i in 0..5_u32 {
        let mut all_zero = true;
        for j in 0..6_u32 {
            let z: bool = *input.at(i).at(j) == 0_u32;
            all_zero = all_zero && z;
        }
        zero_rows.append(all_zero);
    }

    let mut zero_cols = ArrayTrait::new();
    for j in 0..6_u32 {
        let mut all_zero = true;
        for i in 0..5_u32 {
            let z: bool = *input.at(i).at(j) == 0_u32;
            all_zero = all_zero && z;
        }
        zero_cols.append(all_zero);
    }

    let flat_result = array![
        *result.at(0_u32).at(0_u32), *result.at(0_u32).at(1_u32), *result.at(0_u32).at(2_u32), *result.at(0_u32).at(3_u32),
        *result.at(1_u32).at(0_u32), *result.at(1_u32).at(1_u32), *result.at(1_u32).at(2_u32), *result.at(1_u32).at(3_u32),
        *result.at(2_u32).at(0_u32), *result.at(2_u32).at(1_u32), *result.at(2_u32).at(2_u32), *result.at(2_u32).at(3_u32),
    ];

    let mut idx: u32 = 0;
    for i in 0..5_u32 {
        for j in 0..6_u32 {
            if !(*zero_rows.at(i) || *zero_cols.at(j)) {
                let k: u32 = idx;
                assert!(*input.at(i).at(j) == *flat_result.at(k));
                idx += 1_u32;
            }
        }
    }

    assert!(idx == 12_u32);
}