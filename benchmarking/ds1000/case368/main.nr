fn main(a: [[Field; 5]; 3], result: Field) {
    // Compare all elements of a to a[0]
    let mut all_equal: bool = true;

    for i in 0..3 {
        for j in 0..5 {
            let eq = a[i][j] == a[0][j];
            all_equal = all_equal & eq;
        }
    }

    // Verify final result
    assert((result == 1) == all_equal, "Matrix rows not identical");
}
