#[executable]
pub fn main() {
    // A = [1, 2, 3, 4, 5, 6, 7]
    let A = array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32];
    // B = [[1, 2], [3, 4], [5, 6]]
    let B = array![
        array![1_u32, 2_u32],
        array![3_u32, 4_u32],
        array![5_u32, 6_u32],
    ];

    let ncol: u32 = 2_u32;
    let nrow: u32 = 3_u32;

    // truncated = [A[0], A[1], A[2], A[3], A[4], A[5]]
    let truncated = array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32];

    // for i in range(nrow):
    //     for j in range(ncol):
    //         idx = i * ncol + j
    //         assert B[i][j] == truncated[idx]
    for i in 0..nrow {
        for j in 0..ncol {
            let idx: u32 = i * ncol + j;
            assert!(*B.at(i).at(j) == *truncated.at(idx));
        }
    }
}
