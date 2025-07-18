#[executable]
pub fn main() {
    let a = array![array![1_u32, 5_u32, 9_u32, 13_u32], array![2_u32, 6_u32, 10_u32, 14_u32], array![3_u32, 7_u32, 11_u32, 15_u32], array![4_u32, 8_u32, 12_u32, 16_u32]];
    let result = array![array![array![1_u32, 5_u32], array![2_u32, 6_u32]], array![array![9_u32, 13_u32], array![10_u32, 14_u32]], array![array![3_u32, 7_u32], array![4_u32, 8_u32]], array![array![11_u32, 15_u32], array![12_u32, 16_u32]]];
    for k in 0..4_u32 {
        let i_0: u32 = k / 2_u32;
        let j_0: u32 = k % 2_u32;
        for u in 0..2_u32 {
            for v in 0..2_u32 {
                let t1: u32 = i_0 * 2_u32 + u;
                let t2: u32 = j_0 * 2_u32 + v;
                let orig = *a.at(t1).at(t2);
                let out  = *result.at(k).at(u).at(v);
                assert!(orig == out);
            }
        }
    }
}