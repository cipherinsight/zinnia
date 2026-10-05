use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<Vec<u64>> = Vec::new();
    for i in 0..2 {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..3 {
            tmp.push(env::read());
        }
        data.push(tmp);
    }
    let result = env::read();


    let mut answer = 0;
    let mut tmp = 0;
    for i in 0..2 {
        for j in 0..3 {
            if data[i][j] > tmp {
                answer = i * 3 + j;
                tmp = data[i][j];
            }
        }
    }

    assert_eq!(answer as u64, result);

    // write public output to the journal
    // env::commit(&input);
}
