fn main(data: [Field; 3], result: [[Field; 4]; 3]) {
    // Loop over rows
    for i in 0..3 {
        // Loop over columns
        for j in 0..4 {
            // Compare data[i] to the literal j
            if data[i] == Field::from(j as u128) {
                // Must be 1 if j == data[i]
                assert(result[i][j] == 1, "Expected a 1 at position ({i},{j})");
            } else {
                // Otherwise must be 0
                assert(result[i][j] == 0, "Expected a 0 at position ({i},{j})");
            }
        }
    }
}
