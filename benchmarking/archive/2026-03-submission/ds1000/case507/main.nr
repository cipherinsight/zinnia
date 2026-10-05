fn main(a: [Field; 3], result: [[Field; 4]; 3]) {
    // One-hot per row: result[i][j] = 1 iff a[i] == j, else 0
    for i in 0..3 {
        for j in 0..4 {
            let is_match = a[i] == Field::from(j as u128);
            if is_match {
                assert(result[i][j] == 1, "Expected 1 at ({i},{j})");
            } else {
                assert(result[i][j] == 0, "Expected 0 at ({i},{j})");
            }
        }
    }
}
