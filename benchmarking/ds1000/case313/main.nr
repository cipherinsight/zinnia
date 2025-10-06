fn main(a: [[Field; 3]; 2], result: Field) {
    // Flatten 2x3 in C order
    let flat: [Field; 6] = [
        a[0][0], a[0][1], a[0][2],
        a[1][0], a[1][1], a[1][2],
    ];

    // Compute argmax over flat
    let mut max_val = flat[0];
    let mut argmax  = 0;

    for k in 1..6 {
        let cur = flat[k];
        let greater: bool = max_val.lt(cur);
        if greater {
            max_val = cur;
            argmax = Field::from(k as u128);
        }
    }

    assert(result == argmax, "Argmax mismatch: expected {argmax}, got {result}");
}
