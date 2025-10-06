fn main(A: [Field; 7], B: [[Field; 2]; 3]) {
    let ncol = 2;
    let nrow = 3;

    // Truncate the last element: only the first 6 values are used
    let truncated: [Field; 6] = [
        A[0], A[1], A[2],
        A[3], A[4], A[5],
    ];

    // Verify reshape correctness
    for i in 0..nrow {
        for j in 0..ncol {
            let idx = i * ncol + j;
            assert(B[i][j] == truncated[idx], "Mismatch at ({i},{j})");
        }
    }
}
