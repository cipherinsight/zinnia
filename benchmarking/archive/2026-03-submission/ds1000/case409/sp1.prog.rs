// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read x (3x3)
    let mut x: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        x.push(row);
    }

    // read y (3x3)
    let mut y: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        y.push(row);
    }

    // read z (3x3)
    let mut z: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(sp1_zkvm::io::read::<i32>());
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

    // sp1_zkvm::io::commit_slice(&output);
}
