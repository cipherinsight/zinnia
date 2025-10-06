// Noir
fn main(A: [Field; 6], B: [[Field; 2]; 3]) {
    let nrow = 3;
    let ncol = 2;

    for i in 0..nrow {
        for j in 0..ncol {
            // idx = i * ncol + j  (C-order flattening)
            let idx = i * ncol + j;
            assert(B[i][j] == A[idx], "Reshape mismatch at ({i},{j})");
        }
    }
}
