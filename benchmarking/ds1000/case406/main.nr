fn main(a: [Field; 10], index: [Field; 10], result: [Field; 3]) {
    // Compute per-group maxima for groups {0,1,2}.
    let mut max0: Field = 0;
    let mut max1: Field = 0;
    let mut max2: Field = 0;

    for i in 0..10 {
        let v = a[i];

        // Group 0
        let is0: bool = index[i] == Field::from(0);
        if is0 {
            let bigger0: bool = max0.lt(v);
            if bigger0 { max0 = v; }
        }

        // Group 1
        let is1: bool = index[i] == Field::from(1);
        if is1 {
            let bigger1: bool = max1.lt(v);
            if bigger1 { max1 = v; }
        }

        // Group 2
        let is2: bool = index[i] == Field::from(2);
        if is2 {
            let bigger2: bool = max2.lt(v);
            if bigger2 { max2 = v; }
        }
    }

    // Verify the resulting vector matches [max0, max1, max2]
    assert(result[0] == max0, "group-0 max mismatch");
    assert(result[1] == max1, "group-1 max mismatch");
    assert(result[2] == max2, "group-2 max mismatch");
}
