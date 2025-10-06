// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a: NDArray[int, 3, 2, 2]
    let a = array![
        array![array![10_u32, 20_u32], array![30_u32, 40_u32]],
        array![array![6_u32, 7_u32], array![8_u32, 9_u32]],
        array![array![10_u32, 11_u32], array![12_u32, 13_u32]],
    ];

    // permutation: NDArray[int, 3]
    let permutation = array![1_u32, 0_u32, 2_u32];

    // result: NDArray[int, 3, 2, 2]
    let result = array![
        array![array![6_u32, 7_u32], array![8_u32, 9_u32]],
        array![array![10_u32, 20_u32], array![30_u32, 40_u32]],
        array![array![10_u32, 11_u32], array![12_u32, 13_u32]],
    ];

    // We want: result[k, r, s] == a[c[k], r, s], where c[k] is the inverse permutation of `permutation`.
    // c[k] = sum_i i * [permutation[i] == k]
    // Then select a[c[k], r, s] by indicator sum over t in 0..3.

    for k in 0..3_u32 {
        // Build inverse index c[k]
        let mut ck: u32 = 0_u32;
        for i in 0..3_u32 {
            let is_target: bool = *permutation.at(i) == k;
            let mut ind_u32: u32 = 0_u32;
            if is_target {
                ind_u32 = 1_u32;
            } else {
                ind_u32 = 0_u32;
            }
            ck = ck + i * ind_u32;
        }

        // For each inner position (r, s), select a[ck, r, s]
        for r in 0..2_u32 {
            for s in 0..2_u32 {
                let mut selected: u32 = 0_u32;
                for t in 0..3_u32 {
                    let eq: bool = ck == t;
                    let mut ind_u32: u32 = 0_u32;
                    if eq {
                        ind_u32 = 1_u32;
                    } else {
                        ind_u32 = 0_u32;
                    }
                    selected = selected + *a.at(t).at(r).at(s) * ind_u32;
                }
                assert!(*result.at(k).at(r).at(s) == selected);
            }
        }
    }
}
