fn pow_field(base_in: Field, exp: Field) -> Field {
    let mut res: Field = 1;
    for i in 0..252 {
        res = res * res;
        let mask: u128 = (((exp as u128) >> ((251 - (i as u8)) as u8))) & 1;
        if mask == 1 {
            res = res * base_in;
        }
    }
    res
}

fn main(a: [[Field; 2]; 2], power: Field, result: [[Field; 2]; 2]) {
    // Elementwise exponentiation: computed = a ** power
    let c00 = pow_field(a[0][0], power);
    let c01 = pow_field(a[0][1], power);
    let c10 = pow_field(a[1][0], power);
    let c11 = pow_field(a[1][1], power);

    assert(result[0][0] == c00, "Mismatch at (0,0)");
    assert(result[0][1] == c01, "Mismatch at (0,1)");
    assert(result[1][0] == c10, "Mismatch at (1,0)");
    assert(result[1][1] == c11, "Mismatch at (1,1)");
}
