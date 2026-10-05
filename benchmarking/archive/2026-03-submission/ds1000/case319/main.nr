fn main(a: [[[Field; 2]; 2]; 3], permutation: [Field; 3], result: [[[Field; 2]; 2]; 3]) {
    // We want: result[k, r, c] == a[c[k], r, c], where
    // c[k] is the inverse permutation of `permutation`.
    //
    // Compute c[k] = sum_i i * [permutation[i] == k]
    // Then select a[c[k], r, c] by indicator sum over t in 0..3.

    for k in 0..3 {
        // Build inverse index c[k] as a Field
        let mut ck: Field = 0;
        for i in 0..3 {
            let is_target: bool = permutation[i] == Field::from(k as u128);
            let ind: Field = if is_target { 1 } else { 0 };
            ck = ck + Field::from(i as u128) * ind;
        }

        // For each inner position (r, s), select a[ck][r][s]
        for r in 0..2 {
            for s in 0..2 {
                let mut selected: Field = 0;
                for t in 0..3 {
                    let is_eq: bool = ck == Field::from(t as u128);
                    let ind: Field = if is_eq { 1 } else { 0 };
                    selected = selected + a[t][r][s] * ind;
                }
                assert(result[k][r][s] == selected, "Mismatch at (k={k}, r={r}, c={s})");
            }
        }
    }
}
