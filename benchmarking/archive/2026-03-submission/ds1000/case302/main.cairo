// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure preserved.

#[executable]
pub fn main() {
    // A: NDArray[int, 6]
    let A = array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32];

    // B: NDArray[int, 3, 2]
    let B = array![
        array![1_u32, 2_u32],
        array![3_u32, 4_u32],
        array![5_u32, 6_u32],
    ];

    // nrow = 3, ncol = 2
    let nrow: u32 = 3_u32;
    let ncol: u32 = 2_u32;

    // for i in range(nrow):
    //   for j in range(ncol):
    //     idx = i * ncol + j
    //     assert B[i, j] == A[idx]
    for i in 0_u32..nrow {
        for j in 0_u32..ncol {
            let idx: u32 = i * ncol + j;
            assert!(*B.at(i).at(j) == *A.at(idx));
        }
    }
}
