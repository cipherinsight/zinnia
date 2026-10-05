fn main(nums: [Field; 20], result: Field) {
    for t in 0..20 {
        let num = nums[t];
        assert(!num.lt(0), "nums must be >= 0");
        assert(num.lt(20), "nums must be < 20");
    }
    let mut values: [Field; 20] = [0; 20];
    for t in 0..20 {
        let num = nums[t];
        for k in 0..20 {
            if num == Field::from(k as u128) {
                values[k] = values[k] + Field::from(k as u128);
            }
        }
    }
    let mut take: Field = 0;
    let mut skip: Field = 0;
    for i in 0..20 {
        let take_i = skip + values[i];
        let skip_i = if skip.lt(take) { take } else { skip };
        take = take_i;
        skip = skip_i;
    }
    let best = if take.lt(skip) { skip } else { take };
    assert(result == best, "result mismatch");
}