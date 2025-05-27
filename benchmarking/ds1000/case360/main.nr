fn main(
    data: [[Field; 4]; 3],
    result: [[Field; 4]; 2]
) {
    // Check row 0 and row 1
    for i in 0..2 {
        for j in 0..4 {
            assert(
                data[i][j] == result[i][j],
                "Mismatch at position ({i},{j})"
            );
        }
    }
}