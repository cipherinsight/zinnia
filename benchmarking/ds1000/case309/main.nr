fn main(a: [[Field; 3]; 2], result: Field) {
    let flat: [Field; 6] = [
        a[0][0], a[0][1], a[0][2],
        a[1][0], a[1][1], a[1][2],
    ];

    let mut max_val = flat[0];
    let mut arg    = 0;          // index 0

    for k in 1..6 {
        let cur = flat[k];
        let greater: bool = max_val.lt(cur);

        if greater {
            max_val = cur;
            arg = Field::from((k as u128));
        }
    }

    assert(arg == result, "argmax mismatch: expected {arg}, got {result}");
}
