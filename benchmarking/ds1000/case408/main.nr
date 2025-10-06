fn main(a: [Field; 10], index: [Field; 10], result: [Field; 3]) {
    // -------- First pass: seed minima from first occurrence in each group --------
    let mut found0: bool = false;
    let mut found1: bool = false;
    let mut found2: bool = false;

    let mut min0: Field = 0;
    let mut min1: Field = 0;
    let mut min2: Field = 0;

    for i in 0..10 {
        if !found0 & (index[i] == Field::from(0)) {
            min0 = a[i];
            found0 = true;
        }
        if !found1 & (index[i] == Field::from(1)) {
            min1 = a[i];
            found1 = true;
        }
        if !found2 & (index[i] == Field::from(2)) {
            min2 = a[i];
            found2 = true;
        }
    }

    // Each of the groups {0,1,2} must appear at least once
    assert(found0 & found1 & found2, "One of the groups is empty");

    // -------- Second pass: refine minima --------
    for i in 0..10 {
        if index[i] == Field::from(0) {
            if a[i].lt(min0) { min0 = a[i]; }
        }
        if index[i] == Field::from(1) {
            if a[i].lt(min1) { min1 = a[i]; }
        }
        if index[i] == Field::from(2) {
            if a[i].lt(min2) { min2 = a[i]; }
        }
        // labels not in {0,1,2} are ignored
    }

    // Compare against expected
    assert(result[0] == min0, "group-0 min mismatch");
    assert(result[1] == min1, "group-1 min mismatch");
    assert(result[2] == min2, "group-2 min mismatch");
}
