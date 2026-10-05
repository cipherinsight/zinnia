#[executable]
pub fn main() {
    let matrix = array![array![5_u32, 0_u32, 3_u32, 3_u32, 7_u32, 9_u32, 3_u32, 5_u32, 2_u32, 4_u32], array![7_u32, 6_u32, 8_u32, 8_u32, 1_u32, 6_u32, 7_u32, 7_u32, 8_u32, 1_u32], array![5_u32, 9_u32, 8_u32, 9_u32, 4_u32, 3_u32, 0_u32, 3_u32, 5_u32, 0_u32], array![2_u32, 3_u32, 8_u32, 1_u32, 3_u32, 3_u32, 3_u32, 7_u32, 0_u32, 1_u32], array![9_u32, 9_u32, 0_u32, 4_u32, 7_u32, 3_u32, 2_u32, 7_u32, 2_u32, 0_u32], array![0_u32, 4_u32, 5_u32, 5_u32, 6_u32, 8_u32, 4_u32, 1_u32, 4_u32, 9_u32], array![8_u32, 1_u32, 1_u32, 7_u32, 9_u32, 9_u32, 3_u32, 6_u32, 7_u32, 2_u32], array![0_u32, 3_u32, 5_u32, 9_u32, 4_u32, 4_u32, 6_u32, 4_u32, 4_u32, 3_u32], array![4_u32, 4_u32, 8_u32, 4_u32, 3_u32, 7_u32, 5_u32, 5_u32, 0_u32, 1_u32], array![5_u32, 9_u32, 3_u32, 0_u32, 5_u32, 0_u32, 1_u32, 2_u32, 4_u32, 2_u32]];
    let valid = 0_u32;

    assert!(valid >= 0);
    assert!(valid < 2_u32);
    for i in 0..10_u32 {
        for j in 0..10_u32 {
            let x = *matrix.at(i).at(j);
            if valid == 1_u32 {
                assert!(1_u32 < x);
                assert!(10_u32 >= x);
            }
        }
    }
}