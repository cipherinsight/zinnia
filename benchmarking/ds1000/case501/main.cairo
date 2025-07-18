#[executable]
pub fn main() {
    let a = array![array![array![0_u32, 1_u32], array![2_u32, 3_u32], array![4_u32, 5_u32]], array![array![6_u32, 7_u32], array![8_u32, 9_u32], array![10_u32, 11_u32]], array![array![12_u32, 13_u32], array![14_u32, 15_u32], array![16_u32, 17_u32]]];
    let b = array![array![0_u32, 1_u32, 1_u32], array![1_u32, 0_u32, 1_u32], array![1_u32, 1_u32, 0_u32]];
    let desired = array![array![0_u32, 3_u32, 5_u32], array![7_u32, 8_u32, 11_u32], array![13_u32, 15_u32, 16_u32]];


    for i in 0..3_u32 {
        for j in 0..3_u32 {
            let bij: u32 = *b.at(i).at(j);
            assert!((bij == 0_u32) || (bij == 1_u32));

            if bij == 0_u32 {
                assert!(*a.at(i).at(j).at(0) == *desired.at(i).at(j));
            } else {
                assert!(*a.at(i).at(j).at(1) == *desired.at(i).at(j));
            }
        }
    }
}