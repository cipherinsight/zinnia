fn main(a: [Field; 10], accmap: [Field; 10], result: [Field; 3]) {
    // add = 2  => add + 1 = 3 (only used if we encountered -1)
    let add_plus_one: u32 = 3;

    // Accumulate sums for groups {0,1,2}
    let mut sum0: Field = 0;
    let mut sum1: Field = 0;
    let mut sum2: Field = 0;

    for i in 0..10 {
        let adjusted: Field = accmap[i];
        let v = a[i];

        if adjusted == Field::from(0) { sum0 = sum0 + v; }
        if adjusted == Field::from(1) { sum1 = sum1 + v; }
        if adjusted == Field::from(2) { sum2 = sum2 + v; }
        // Values mapped to other buckets (e.g., 6 or 8) are ignored.
    }

    // Check expected = [sum0, sum1, sum2]
    assert(result[0] == sum0, "group-0 sum mismatch");
    assert(result[1] == sum1, "group-1 sum mismatch");
    assert(result[2] == sum2, "group-2 sum mismatch");
}
