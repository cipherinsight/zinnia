fn main(matrix: [[Field; 10]; 8], sol:   [[Field; 10]; 8]) {
    for i in 0..8 {
        for j in 0..10 {
            if matrix[i][j] == 0 {
                for k in 0..8 {
                    assert(sol[k][j] == 0, "sol must be 0 when matrix is 0");
                }
                for k in 0..10 {
                    assert(sol[i][k] == 0, "sol must be 0 when matrix is 0");
                }
            }
        }
    }
}