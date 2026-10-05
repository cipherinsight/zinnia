#[executable]
pub fn main() {
    // Dynamic-shape style task with bounded sample payload.
    let data = array![1_u32, 0_u32, 3_u32];
    let result = array![
        array![0_u32, 1_u32, 0_u32, 0_u32],
        array![1_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 1_u32],
    ];

    let mut max_value: u32 = *data.at(0_u32);
    for i in 1_u32..3_u32 {
        let v: u32 = *data.at(i);
        if v > max_value {
            max_value = v;
        }
    }

    // For this benchmark witness, max(a)+1 is 4.
    assert!(max_value + 1_u32 == 4_u32);

    for i in 0_u32..3_u32 {
        for j in 0_u32..4_u32 {
            if j == *data.at(i) {
                assert!(*result.at(i).at(j) == 1_u32);
            } else {
                assert!(*result.at(i).at(j) == 0_u32);
            }
        }
    }
}