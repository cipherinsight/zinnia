// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read x (3x3)
    let mut x: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read());
        }
        x.push(row);
    }

    // read y (3x3)
    let mut y: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read());
        }
        y.push(row);
    }

    // read z (3x3)
    let mut z: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read());
        }
        z.push(row);
    }

    // compute expected = x + y
    let mut expected: Vec<Vec<i32>> = Vec::new();
    for i in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for j in 0..3 {
            row.push(x[i][j as usize] + y[i][j as usize]);
        }
        expected.push(row);
    }

    // assert z == expected
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(z[i][j as usize], expected[i][j as usize]);
        }
    }

    // env::commit(&output);
}
