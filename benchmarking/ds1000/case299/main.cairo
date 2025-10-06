#[executable]
pub fn main() {
    // a: NDArray[int, 2, 3]
    let a = array![
        array![1_u32, 0_u32, 3_u32],
        array![2_u32, 4_u32, 1_u32],
    ];

    // result: NDArray[int, 6, 5]
    let result = array![
        array![0_u32, 1_u32, 0_u32, 0_u32, 0_u32], // 1 → index 1
        array![1_u32, 0_u32, 0_u32, 0_u32, 0_u32], // 0 → index 0
        array![0_u32, 0_u32, 0_u32, 1_u32, 0_u32], // 3 → index 3
        array![0_u32, 0_u32, 1_u32, 0_u32, 0_u32], // 2 → index 2
        array![0_u32, 0_u32, 0_u32, 0_u32, 1_u32], // 4 → index 4
        array![0_u32, 1_u32, 0_u32, 0_u32, 0_u32], // 1 → index 1
    ];

    // a_min = 0
    let a_min: u32 = 0_u32;

    // flat = [a[0,0], a[0,1], a[0,2], a[1,0], a[1,1], a[1,2]]
    let flat = array![
        *a.at(0_u32).at(0_u32),
        *a.at(0_u32).at(1_u32),
        *a.at(0_u32).at(2_u32),
        *a.at(1_u32).at(0_u32),
        *a.at(1_u32).at(1_u32),
        *a.at(1_u32).at(2_u32),
    ];

    // for i in range(6):
    //   for j in range(5):
    //     expected = 1 if (flat[i] - a_min) == j else 0
    //     assert result[i, j] == expected
    for i in 0_u32..6_u32 {
        let diff: u32 = *flat.at(i) - a_min;
        for j in 0_u32..5_u32 {
            if diff == j {
                assert!(*result.at(i).at(j) == 1_u32);
            } else {
                assert!(*result.at(i).at(j) == 0_u32);
            }
        }
    }
}
