unconstrained fn mydivision(a: Field, b: Field) -> (Field, Field) {
    let aa = a as u128;
    let bb = b as u128;
    let reminder = aa % bb;
    let ra: Field = Field::from((aa - reminder) / bb);
    let rb: Field = Field::from(reminder);
    (ra, rb)
}

fn main(
    area: Field,
    expected_l: Field,
    expected_w: Field
) {
    let mut w: Field = area;
    let mut done: bool = false;

    for i in 1..1001 {
        let fi = Field::from(i as u128);
        if !done {
            let (q, r) = unsafe {mydivision(area, fi)};
            assert(q * fi + r == area);
            let divisible: bool = (fi * q) == area;
            if divisible {
                w = fi;
            }
            let isq: Field = Field::from((i * i) as u128);
            let breakcond: bool = !isq.lt(area);
            if breakcond {
                done = true;
            }
        }
    }

    let answer_w = w;
    let answer_l = area / w;

    if answer_l.lt(answer_w) {
        let tmp = answer_l;
        let answer_l = answer_w;
        let answer_w = tmp;
        assert(answer_l == expected_l, "L mismatch");
        assert(answer_w == expected_w, "W mismatch");
    } else {
        assert(answer_l == expected_l, "L mismatch");
        assert(answer_w == expected_w, "W mismatch");
    }
}
