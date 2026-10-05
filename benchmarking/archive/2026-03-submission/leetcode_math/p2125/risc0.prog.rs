use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let m = 5;
    let n = 5;
    let mut res = 0;

    let mut bank: Vec<Vec<i32>> = Vec::new();
    for i in 0..5 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..5 {
            tmp.push(env::read());
        }
        bank.push(tmp);
    }
    let expected = env::read();

    for si in 0..n {
        for sj in 0..m {
            for ti in 0..n {
                for tj in 0..m {
                    let mut add_one = bank[si][sj] == 1 && bank[ti][tj] == 1 && si < ti;

                    for k in (si + 1)..ti {
                        if (sj..tj).any(|j| bank[k][j] == 1) {
                            add_one = false;
                            break;
                        }
                    }

                    if add_one {
                        res += 1;
                    }
                }
            }
        }
    }

    assert!(res == expected, "Expected {}, but got {}", expected, res);

    // write public output to the journal
    // env::commit(&input);
}
