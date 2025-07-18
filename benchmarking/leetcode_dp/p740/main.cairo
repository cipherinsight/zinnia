#[executable]
pub fn main() {
    let nums = array![13_u32, 16_u32, 1_u32, 4_u32, 4_u32, 8_u32, 10_u32, 19_u32, 5_u32, 7_u32, 13_u32, 2_u32, 7_u32, 8_u32, 15_u32, 18_u32, 6_u32, 14_u32, 9_u32, 10_u32];
    let result = 113_u32;

    for t in 0..20_u32 {
        let num = *nums.at(t);
        assert!(num>=0);
        assert!(num<20);
    }
    let mut values = ArrayTrait::new();
    for i in 0..20_u32 {
        values.append(0_u32);
    }
    for t in 0..20_u32 {
        let num = *nums.at(t);
        for k in 0..20_u32 {
            if num == k {
                let tmp = values.pop_front().unwrap();
                values.append(tmp + k);
            } else {
                let tmp = values.pop_front().unwrap();
                values.append(tmp);
            }
        }
    }
    let mut take = 0_u32;
    let mut skip = 0_u32;
    for i in 0..20_u32 {
        let take_i: u32 = skip + *values.at(i);
        let mut skip_i = skip;
        if skip < take {
            skip_i = take;
        }
        take = take_i;
        skip = skip_i;
    }
    let mut best = 0_u32;
    if take < skip {
        best = skip;
    } else {
        best = take;
    }
    assert!(result == best);
}