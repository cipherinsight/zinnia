fn main(A: [Field; 6], B: [[Field; 2]; 3]) {
    let nrow = 3;
    let ncol = 2;

    // Check that B is a correct reshape of A into shape (3, 2)
    for i in 0..nrow {
        for j in 0..ncol {
            let idx = i * ncol + j;
            assert(B[i][j] == A[idx], "Reshape mismatch at ({i},{j})");
        }
    }
}
