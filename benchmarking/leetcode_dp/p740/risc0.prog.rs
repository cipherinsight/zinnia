use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let n = 20;
    let result: u32 = env::read();
    let mut nums = vec![0; n as usize];
    for i in 0..n {
        let tmp: u32 = env::read();
        nums.push(tmp);
    }

    let mut values = vec![0; n as usize];
    // Populate values array
    for num in nums {
        values[num as usize] += num as i32;
    }

    let mut take = 0;
    let mut skip = 0;

    // Compute the maximum sum with non-adjacent selections
    for i in 0..n {
        let take_i = skip + values[i];
        let skip_i = take.max(skip);
        take = take_i;
        skip = skip_i;
    }

    assert_eq!(
        result,
        take.max(skip) as u32,
        "The computed result does not match the expected result."
    );

    // write public output to the journal
    // env::commit(&input);
}
