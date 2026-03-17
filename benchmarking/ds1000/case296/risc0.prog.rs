use risc0_zkvm::guest::env;

fn main() {
    let rows: usize = env::read::<i32>() as usize;
    let cols: usize = env::read::<i32>() as usize;
    assert!(rows > 0);
    assert!(cols > 0);

    let mut data: Vec<i32> = Vec::new();
    for _ in 0..rows {
        data.push(env::read());
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..rows {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..cols {
            tmp.push(env::read());
        }
        result.push(tmp);
    }

    let mut max_value = data[0];
    for v in &data {
        if *v > max_value {
            max_value = *v;
        }
    }
    assert_eq!(cols, (max_value + 1) as usize);

    for row in &result {
        assert_eq!(row.len(), cols);
    }

    for i in 0..rows {
        for j in 0..cols {
            if (j as i32) == data[i] {
                assert_eq!(result[i][j], 1);
            } else {
                assert_eq!(result[i][j], 0);
            }
        }
    }

    // No public outputs for this verifier-style task.
}
