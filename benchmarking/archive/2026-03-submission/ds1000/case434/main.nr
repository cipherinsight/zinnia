fn main(a: [[Field; 4]; 4], result: [[Field; 4]; 4]) {
    // zero_rows = [1, 3]
    // zero_cols = [1, 2]

    for i in 0..4 {
        for j in 0..4 {
            // Cell is zeroed if its row in {1,3} or its col in {1,2}
            let is_zero_row = (i == 1) | (i == 3);
            let is_zero_col = (j == 1) | (j == 2);

            let should_zero = is_zero_row | is_zero_col;
            let expected = if should_zero { Field::from(0) } else { a[i][j] };

            assert(result[i][j] == expected, "Mismatch at ({i},{j})");
        }
    }
}
