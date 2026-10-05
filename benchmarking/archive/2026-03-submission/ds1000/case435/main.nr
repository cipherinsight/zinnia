fn main(a: [[Field; 4]; 4], result: [[Field; 4]; 4]) {
    // Zero out row 1 and column 0
    for i in 0..4 {
        for j in 0..4 {
            let should_zero = (i == 1) | (j == 0);
            let expected = if should_zero { Field::from(0) } else { a[i][j] };
            assert(result[i][j] == expected, "Mismatch at ({i},{j})");
        }
    }
}
