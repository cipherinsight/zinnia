// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure preserved.

#[executable]
pub fn main() {
    // a: NDArray[int, 3]
    let a = array![1_u32, 2_u32, 5_u32];

    // result: NDArray[int, 3, 5]
    let result = array![
        array![1_u32, 0_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 1_u32, 0_u32, 0_u32, 0_u32],
        array![0_u32, 0_u32, 0_u32, 0_u32, 1_u32],
    ];

    // a_min = min(a)
    let mut a_min: u32 = *a.at(0_u32);
    for i in 1_u32..3_u32 {
        let v: u32 = *a.at(i);
        if v < a_min {
            a_min = v;
        }
    }

    // for i in range(result.shape[0]):
    //   for j in range(result.shape[1]):
    //     expected = 1 if (a[i] - a_min) == j else 0
    //     assert result[i, j] == expected
    for i in 0_u32..3_u32 {
        let ai_minus: u32 = *a.at(i) - a_min;
        for j in 0_u32..5_u32 {
            if ai_minus == j {
                assert!(*result.at(i).at(j) == 1_u32);
            } else {
                assert!(*result.at(i).at(j) == 0_u32);
            }
        }
    }
}
