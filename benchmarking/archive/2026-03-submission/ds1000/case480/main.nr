fn main(x: [Field; 9], y: [Field; 9], a: Field, b: Field, result: Field) {
    // We search for the first index i where (x[i] == a & y[i] == b).
    // If none found, found_index stays -1.
    // Note: represent -1 as Field::from(-1) for consistency.

    let mut found_index: Field = Field::from(-1);

    for i in 0..9 {
        let match_x = x[i] == a;
        let match_y = y[i] == b;
        let not_found = found_index == Field::from(-1);
        if match_x & match_y & not_found {
            found_index = Field::from(i as u128);
        }
    }

    assert(result == found_index, "First-match index mismatch");
}
