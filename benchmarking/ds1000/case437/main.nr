fn main(a: [[Field; 2]; 3], mask: [[Field; 2]; 3]) {
    // For each row, find the minimum and mark 1 where element == row_min, else 0.
    for i in 0..3 {
        // Row minimum over 2 columns
        let mut row_min = a[i][0];
        if a[i][1].lt(row_min) {
            row_min = a[i][1];
        }

        // Check mask equals (a == row_min)
        for j in 0..2 {
            let expected = if a[i][j] == row_min { Field::from(1) } else { Field::from(0) };
            assert(mask[i][j] == expected, "Mismatch at ({i},{j})");
        }
    }
}
