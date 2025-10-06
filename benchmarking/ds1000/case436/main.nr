fn main(a: [[Field; 2]; 3], mask: [[Field; 2]; 3]) {
    // For each row, find max and mark 1 where element == row_max, else 0.

    for i in 0..3 {
        // Find the row maximum
        let mut row_max = a[i][0];
        if row_max.lt(a[i][1]) {
            row_max = a[i][1];
        }

        // Verify mask entries
        for j in 0..2 {
            let expected = if a[i][j] == row_max { Field::from(1) } else { Field::from(0) };
            assert(mask[i][j] == expected, "Mismatch at ({i},{j})");
        }
    }
}
