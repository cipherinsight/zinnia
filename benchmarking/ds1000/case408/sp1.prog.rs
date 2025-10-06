// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read a
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..10 {
        a.push(sp1_zkvm::io::read::<i32>());
    }

    // read index
    let mut index: Vec<i32> = Vec::new();
    for _ in 0..10 {
        index.push(sp1_zkvm::io::read::<i32>());
    }

    // read result
    let mut result: Vec<i32> = Vec::new();
    for _ in 0..3 {
        result.push(sp1_zkvm::io::read::<i32>());
    }

    let n: usize = a.len();

    // first pass: seed minima
    let mut found0: bool = false;
    let mut found1: bool = false;
    let mut found2: bool = false;
    let mut min0: i32 = 0;
    let mut min1: i32 = 0;
    let mut min2: i32 = 0;

    for i in 0..n {
        if index[i] == 0 && !found0 {
            min0 = a[i];
            found0 = true;
        }
        if index[i] == 1 && !found1 {
            min1 = a[i];
            found1 = true;
        }
        if index[i] == 2 && !found2 {
            min2 = a[i];
            found2 = true;
        }
    }

    assert!(found0 && found1 && found2);

    // second pass: refine minima
    for i in 0..n {
        if index[i] == 0 && a[i] < min0 {
            min0 = a[i];
        }
        if index[i] == 1 && a[i] < min1 {
            min1 = a[i];
        }
        if index[i] == 2 && a[i] < min2 {
            min2 = a[i];
        }
    }

    let expected: Vec<i32> = vec![min0, min1, min2];
    for j in 0..3 {
        assert_eq!(result[j as usize], expected[j as usize]);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
