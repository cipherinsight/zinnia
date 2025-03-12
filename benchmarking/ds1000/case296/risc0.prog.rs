use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<i32> = Vec::new();
    for i in 0..3 {
        data.push(env::read());
    }
    let mut result: Vec<Vec<i32>> = Vec::new();
    for i in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..4 {
            tmp.push(env::read());
        }
        result.push(tmp);
    }


    for i in 0..3 {
        for j in 0..4 {
            if j == data[i] {
                assert_eq!(result[i][j as usize], 1);
            } else {
                assert_eq!(result[i][j as usize], 0);
            }
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
