fn main(a: [Field; 3], result: [[Field; 5]; 3]) {
    // a_min = min(a)
    let mut a_min = a[0];
    if a[1].lt(a_min) { a_min = a[1]; }
    if a[2].lt(a_min) { a_min = a[2]; }

    // One-hot over (a[i] - a_min) in range 0..5
    for i in 0..3 {
        for j in 0..5 {
            let shifted = a[i] - a_min; // e.g., [0,1,4]
            let is_match = shifted == Field::from(j as u128);
            if is_match {
                assert(result[i][j] == 1, "Expected 1 at ({i},{j})");
            } else {
                assert(result[i][j] == 0, "Expected 0 at ({i},{j})");
            }
        }
    }
}
