fn main(a: [[Field; 3]; 2], result: [[Field; 5]; 6]) {
    // a_min = 0
    let a_min: Field = Field::from(0);

    // Flatten the 2x3 matrix in C order
    let flat: [Field; 6] = [
        a[0][0], a[0][1], a[0][2],
        a[1][0], a[1][1], a[1][2],
    ];

    // For each flat element i, and each class j, ensure one-hot encoding
    for i in 0..6 {
        for j in 0..5 {
            let shifted = flat[i] - a_min;
            let is_match = shifted == Field::from(j as u128);
            if is_match {
                assert(result[i][j] == 1, "Expected 1 at ({i},{j})");
            } else {
                assert(result[i][j] == 0, "Expected 0 at ({i},{j})");
            }
        }
    }
}
