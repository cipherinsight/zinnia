#[executable]
pub fn main() {
    let data = array![
        array![10_u32, 50_u32, 30_u32], array![60_u32, 20_u32, 40_u32]
    ];
    let result = 3_u32;

    let mut answer = 0;
    let mut tmp = 0;
    for i in 0..2_u32 {
        for j in 0..3_u32 {
            if *data.at(i).at(j) > tmp {
                answer = i * 3 + j;
                tmp = *data.at(i).at(j);
            }
        }
    }

    assert!(answer == result);
}