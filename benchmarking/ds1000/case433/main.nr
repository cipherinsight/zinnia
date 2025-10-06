fn main(a: [[Field; 4]; 4], result: [[Field; 4]; 4]) {
    // zero_rows = 0, zero_cols = 0
    for i in 0..4 {
        for j in 0..4 {
            // If we're on row 0 or col 0, value should be 0; otherwise keep original.
            let expected = if (i == 0) | (j == 0) {
                Field::from(0)
            } else {
                a[i][j]
            };
            assert(result[i][j] == expected, "Mismatch at ({i},{j})");
        }
    }
}
