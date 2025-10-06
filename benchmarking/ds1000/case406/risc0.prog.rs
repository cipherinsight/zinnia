// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read a
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..10 {
        a.push(env::read());
    }

    // read index
    let mut index: Vec<i32> = Vec::new();
    for _ in 0..10 {
        index.push(env::read());
    }

    // read result
    let mut result: Vec<i32> = Vec::new();
    for _ in 0..3 {
        result.push(env::read());
    }

    // compute expected = [max0, max1, max2]
    let mut max0: i32 = 0;
    let mut max1: i32 = 0;
    let mut max2: i32 = 0;

    for i in 0..10 {
        if index[i as usize] == 0 && a[i as usize] > max0 {
            max0 = a[i as usize];
        }
        if index[i as usize] == 1 && a[i as usize] > max1 {
            max1 = a[i as usize];
        }
        if index[i as usize] == 2 && a[i as usize] > max2 {
            max2 = a[i as usize];
        }
    }

    let expected: Vec<i32> = vec![max0, max1, max2];

    for j in 0..3 {
        assert_eq!(result[j as usize], expected[j as usize]);
    }

    // env::commit(&output);
}
