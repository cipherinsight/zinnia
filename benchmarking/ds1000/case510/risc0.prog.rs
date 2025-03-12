use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut input: Vec<Vec<u64>> = Vec::new();
    for i in 0..5 {
        let mut tmp1: Vec<u64> = Vec::new();
        for j in 0..6 {
            tmp1.push(env::read());
        }
        input.push(tmp1);
    }
    let mut result: Vec<Vec<u64>> = Vec::new();
    for i in 0..3 {
        let mut tmp1: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp1.push(env::read());
        }
        result.push(tmp1);
    }

    let mut zero_rows = vec![true; 5];
    let mut zero_cols = vec![true; 6];

    // Check for zero rows
    for i in 0..5 {
        zero_rows[i] = input[i].iter().all(|&value| value == 0);
    }

    // Check for zero columns
    for j in 0..6 {
        zero_cols[j] = (0..5).all(|i| input[i][j] == 0);
    }

    let flatten_result: Vec<u64> = result.iter().flat_map(|r| r.iter()).cloned().collect();
    let mut idx = 0;

    for i in 0..5 {
        for j in 0..6 {
            if zero_rows[i] || zero_cols[j] {
                continue;
            }
            assert_eq!(flatten_result[idx], input[i][j]);
            idx += 1;
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
