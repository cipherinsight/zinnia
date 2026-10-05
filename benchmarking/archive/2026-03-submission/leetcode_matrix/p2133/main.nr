fn main(matrix: [[Field; 10]; 10], valid: Field) {
    assert(!valid.lt(0), "valid must be >= 0");
    assert(valid.lt(2),  "valid must be <= 1");
    for i in 0..10 {
        for j in 0..10 {
            let x = matrix[i][j];
            if valid == 1 {
                assert(1.lt(x), "matrix must be > 1 when valid = 1");
                assert(!10.lt(x), "matrix must be ≤ 10 when valid = 1");
            }
        }
    }
}
