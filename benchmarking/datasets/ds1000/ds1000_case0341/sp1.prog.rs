// DS-1000 case 341 — SP1 guest.
// `X.flat` on Integer[5, 6] — emits 30 ints in row-major order.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut x: Vec<Vec<i64>> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut row: Vec<i64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(sp1_zkvm::io::read::<i64>());
        }
        x.push(row);
    }
    let mut claimed: Vec<i64> = Vec::with_capacity(30);
    for _ in 0..30 {
        claimed.push(sp1_zkvm::io::read::<i64>());
    }

    let mut result: Vec<i64> = Vec::with_capacity(30);
    for i in 0..5usize {
        for j in 0..6usize {
            result.push(x[i][j]);
        }
    }

    for k in 0..30usize {
        assert_eq!(result[k], claimed[k]);
    }
}
