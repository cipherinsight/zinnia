fn main(a: [[Field; 5]; 2], permutation: [Field; 5], result: [[Field; 5]; 2]) {
    // Goal: permuted[:, j] = a[:, c[j]], where c is the inverse permutation of `permutation`.
    // We construct c[j] as a Field value using one-hot indicators:
    //   c[j] = sum_i i * [permutation[i] == j]
    // and then select a[:, c[j]] using another indicator sum over t in 0..5.

    // For each output column j
    for j in 0..5 {
        // Build c[j] as Field via equality indicators
        let mut cj: Field = 0;
        for i in 0..5 {
            let is_target: bool = permutation[i] == Field::from(j as u128);
            let ind: Field = if is_target { 1 } else { 0 };
            cj = cj + Field::from(i as u128) * ind;
        }

        // For each row r, select a[r][cj] by summing over all possible t with indicator [cj == t]
        // Row 0
        let mut sel_val_r0: Field = 0;
        for t in 0..5 {
            let is_eq: bool = cj == Field::from(t as u128);
            let ind: Field = if is_eq { 1 } else { 0 };
            sel_val_r0 = sel_val_r0 + a[0][t] * ind;
        }
        assert(result[0][j] == sel_val_r0, "Mismatch at (row 0, col {j})");

        // Row 1
        let mut sel_val_r1: Field = 0;
        for t in 0..5 {
            let is_eq: bool = cj == Field::from(t as u128);
            let ind: Field = if is_eq { 1 } else { 0 };
            sel_val_r1 = sel_val_r1 + a[1][t] * ind;
        }
        assert(result[1][j] == sel_val_r1, "Mismatch at (row 1, col {j})");
    }
}
