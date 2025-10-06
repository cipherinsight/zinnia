fn main(a: [[Field; 3]; 2], result: Field) {
    // Flatten the 2x3 matrix in C order
    let flat: [Field; 6] = [
        a[0][0], a[0][1], a[0][2],
        a[1][0], a[1][1], a[1][2],
    ];

    // Find the index of the minimum value
    let mut min_val = flat[0];
    let mut argmin = 0;

    for k in 1..6 {
        let cur = flat[k];
        let smaller: bool = cur.lt(min_val);
        if smaller {
            min_val = cur;
            argmin = Field::from(k as u128);
        }
    }

    // Verify result equals computed argmin
    assert(result == argmin, "Argmin mismatch: expected {argmin}, got {result}");
}
