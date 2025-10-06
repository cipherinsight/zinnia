// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read input A
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..7 {
        a.push(sp1_zkvm::io::read::<i32>());
    }

    // read input B
    let mut b: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..2 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        b.push(tmp);
    }

    let ncol: usize = 2;
    let nrow: usize = 3;
    let truncated: Vec<i32> = vec![a[0], a[1], a[2], a[3], a[4], a[5]];

    for i in 0..nrow {
        for j in 0..ncol {
            let idx: usize = i * ncol + j;
            assert_eq!(b[i][j as usize], truncated[idx]);
        }
    }

    // sp1_zkvm::io::commit_slice(&output);
}
