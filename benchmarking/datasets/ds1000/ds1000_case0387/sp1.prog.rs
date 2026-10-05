// DS-1000 case 387 — SP1 guest program.
// Kernel: a.reshape(50, 2, 100, 2).swapaxes(1, 2).reshape(-1, 2, 2)
// Output[k][p][q] = a[r_block*2 + p][c_block*2 + q] with k = r_block*100 + c_block.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const H: usize = 100;
    const W: usize = 200;
    const R_BLOCKS: usize = 50;
    const C_BLOCKS: usize = 100;
    const K: usize = R_BLOCKS * C_BLOCKS;

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for _ in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for _ in 0..W { row.push(sp1_zkvm::io::read::<f64>()); }
        a.push(row);
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(K * 4);
    for _ in 0..(K * 4) { claimed.push(sp1_zkvm::io::read::<f64>()); }

    let mut result: Vec<f64> = Vec::with_capacity(K * 4);
    for r_block in 0..R_BLOCKS {
        for c_block in 0..C_BLOCKS {
            for p in 0..2 {
                for q in 0..2 {
                    result.push(a[r_block * 2 + p][c_block * 2 + q]);
                }
            }
        }
    }

    for i in 0..(K * 4) {
        assert!((result[i] - claimed[i]).abs() < 1e-9);
    }
}
