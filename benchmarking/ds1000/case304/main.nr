fn main(A: [Field; 7], B: [[Field; 2]; 3]) {
    // We want:
    // truncated = A[1..] = [A[1], A[2], A[3], A[4], A[5], A[6]]
    // reversed  = [A[6], A[5], A[4], A[3], A[2], A[1]]
    // reshaped  = [[A[6], A[5]],
    //              [A[4], A[3]],
    //              [A[2], A[1]]]

    let nrow = 3;
    let ncol = 2;

    for i in 0..nrow {
        for j in 0..ncol {
            // k is the index into the reversed vector of length 6
            let k = i * ncol + j;       // 0..5
            // reversed[k] corresponds to A[6 - k]
            assert(
                B[i][j] == A[6 - k],
                "Mismatch at ({i},{j})"
            );
        }
    }
}
