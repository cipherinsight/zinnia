use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut a: Vec<Vec<Vec<u64>>> = Vec::new();
    for i in 0..3 {
        let mut tmp1: Vec<Vec<u64>> = Vec::new();
        for j in 0..3 {
            let mut tmp2: Vec<u64> = Vec::new();
            for k in 0..2 {
                tmp2.push(env::read());
            }
            tmp1.push(tmp2);
        }
        a.push(tmp1);
    }
    let mut b: Vec<Vec<u64>> = Vec::new();
    for i in 0..3 {
        let mut tmp1: Vec<u64> = Vec::new();
        for j in 0..3 {
            tmp1.push(env::read());
        }
        b.push(tmp1);
    }
    let mut desired: Vec<Vec<u64>> = Vec::new();
    for i in 0..3 {
        let mut tmp1: Vec<u64> = Vec::new();
        for j in 0..3 {
            tmp1.push(env::read());
        }
        desired.push(tmp1);
    }

    for i in 0..3 {
        for j in 0..3 {
            assert!(b[i][j] == 0 || b[i][j] == 1);
            if b[i][j] == 0 {
                assert_eq!(a[i][j][0], desired[i][j]);
            } else if b[i][j] == 1 {
                assert_eq!(a[i][j][1], desired[i][j]);
            }
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
