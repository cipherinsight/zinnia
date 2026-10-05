use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<Vec<f64>> = Vec::new();
    for i in 0..5 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..4 {
            tmp.push(env::read());
        }
        data.push(tmp);
    }
    let mut results: Vec<Vec<f64>> = Vec::new();
    for i in 0..5 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..4 {
            tmp.push(env::read());
        }
        results.push(tmp);
    }

    let mut a = data.clone();
    for i in 0..5 {
        for j in 0..4 {
            let p = (data[i][j] * data[i][j]);
            a[i][j] = p;
        }
    }

    let mut sum_each_row = vec![0.0; 5];
    for i in 0..5 {
        let mut tmp = 0.0;
        for j in 0..4 {
            tmp += a[i][j];
        }
        sum_each_row[i] = tmp.sqrt();
    }

    for i in 0..5 {
        for j in 0..4 {
            assert_eq!(results[i][j], data[i][j] / sum_each_row[i]);
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
