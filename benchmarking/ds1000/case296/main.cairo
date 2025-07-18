#[executable]
pub fn main() {
    let data = array![1_u32, 0_u32, 3_u32];
    let result = array![array![0_u32, 1_u32, 0_u32, 0_u32], array![1_u32, 0_u32, 0_u32, 0_u32], array![0_u32, 0_u32, 0_u32, 1_u32]];
    for i in 0..2_u32 {
        for j in 0..3_u32 {
            if j == *data.at(i) {
                assert!(*result.at(i).at(j) == 1_u32);
            } else {
                assert!(*result.at(i).at(j) == 0_u32);
            }
        }
    }
}