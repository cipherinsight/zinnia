fn main(a: [[Field; 5]; 3], result: Field) {
    // Check if all rows of 'a' are identical to the first row.
    let mut all_equal: bool = true;

    for i in 0..3 {
        for j in 0..5 {
            let eq = a[i][j] == a[0][j];
            all_equal = all_equal & eq;
        }
    }

    assert((result == 1) == all_equal, "Rows are not identical");
}
