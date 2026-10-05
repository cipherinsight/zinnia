#[executable]
pub fn main() {
    let data = array![array![1_u32, 2_u32, 3_u32, 4_u32], array![5_u32, 6_u32, 7_u32, 8_u32], array![9_u32, 10_u32, 11_u32, 12_u32]];
    let result = array![array![1_u32, 2_u32, 3_u32, 4_u32], array![5_u32, 6_u32, 7_u32, 8_u32]];
    for i in 0..2_u32 {
        for j in 0..4_u32 {
            assert!(
                *data.at(i).at(j) == *result.at(i).at(j)
            );
        }
    }
}