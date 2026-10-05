#[executable]
pub fn main() {
    // a: NDArray[int, 3]
    let a = array![1_u32, 0_u32, 3_u32];

    // result: NDArray[int, 3, 4]
    let result = array![
        array![0_u32, 1_u32, 0_u32, 0_u32],
        array![1_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 1_u32],
    ];

    // for i in range(3):
    //   for j in range(4):
    //     expected = 1 if a[i] == j else 0
    //     assert result[i][j] == expected
    for i in 0..3_u32 {
        for j in 0..4_u32 {
            if j == *a.at(i) {
                assert!(*result.at(i).at(j) == 1_u32);
            } else {
                assert!(*result.at(i).at(j) == 0_u32);
            }
        }
    }
}
