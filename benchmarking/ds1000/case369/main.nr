fn main(a: [[Field; 3]; 5], result: Field) {
    // Check that all columns are equal elementwise across each row.
    // Equivalent to: a[i][j] == a[i][0] for all i, j
    let mut all_equal: bool = true;

    for i in 0..5 {
        for j in 0..3 {
            let eq = a[i][j] == a[i][0];
            all_equal = all_equal & eq;
        }
    }

    // Verify the boolean result
    assert((result == 1) == all_equal, "Columns are not all equal elementwise");
}
