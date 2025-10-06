fn main(a: [Field; 4], number: Field, is_contained: Field) {
    // Scan for equality; Noir allows local bools but inputs are Fields.
    let mut found: bool = false;
    for i in 0..4 {
        if a[i] == number {
            found = true;
        }
    }

    // Convert bool -> Field (1 if found else 0)
    let expected: Field = if found { Field::from(1) } else { Field::from(0) };

    assert(is_contained == expected, "containment mismatch");
}
