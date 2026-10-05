// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a: NDArray[int, 2, 5]
    let a = array![
        array![10_u32, 20_u32, 30_u32, 40_u32, 50_u32],
        array![6_u32, 7_u32, 8_u32, 9_u32, 10_u32],
    ];

    // permutation: NDArray[int, 5]
    let permutation = array![0_u32, 4_u32, 1_u32, 3_u32, 2_u32];

    // result: NDArray[int, 2, 5]
    let result = array![
        array![10_u32, 30_u32, 50_u32, 40_u32, 20_u32],
        array![6_u32, 8_u32, 10_u32, 9_u32, 7_u32],
    ];

    // Goal: result[:, j] = a[:, c[j]], where c is the inverse permutation of `permutation`.
    // c[j] = sum_i i * [permutation[i] == j]
    // Then select a[:, c[j]] using indicator sums over t in 0..5.

    for j in 0..5_u32 {
        // Build c[j]
        let mut cj: u32 = 0_u32;
        for i in 0..5_u32 {
            let is_target: bool = *permutation.at(i) == j;
            let mut ind_u32: u32 = 0_u32;
            if is_target {
                ind_u32 = 1_u32;
            } else {
                ind_u32 = 0_u32;
            }
            cj = cj + i * ind_u32;
        }

        // Row 0 selection via indicators
        let mut sel_val_r0: u32 = 0_u32;
        for t in 0..5_u32 {
            let eq: bool = cj == t;
            let mut ind_u32: u32 = 0_u32;
            if eq {
                ind_u32 = 1_u32;
            } else {
                ind_u32 = 0_u32;
            }
            sel_val_r0 = sel_val_r0 + *a.at(0_u32).at(t) * ind_u32;
        }
        assert!(*result.at(0_u32).at(j) == sel_val_r0);

        // Row 1 selection via indicators
        let mut sel_val_r1: u32 = 0_u32;
        for t in 0..5_u32 {
            let eq: bool = cj == t;
            let mut ind_u32: u32 = 0_u32;
            if eq {
                ind_u32 = 1_u32;
            } else {
                ind_u32 = 0_u32;
            }
            sel_val_r1 = sel_val_r1 + *a.at(1_u32).at(t) * ind_u32;
        }
        assert!(*result.at(1_u32).at(j) == sel_val_r1);
    }
}
