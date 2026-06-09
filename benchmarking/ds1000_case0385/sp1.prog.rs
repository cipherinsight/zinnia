// DS-1000 case 385 — SP1 guest program.
// Kernel: a.reshape(50, 2, 100, 2).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
// on Float[100, 200]. Output is (5000, 2, 2) of f64.
//
// Index mapping: for output[k][p][q] with k = c_block*50 + r_block,
//   source = a[r_block*2 + p][c_block*2 + q].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const H: usize = 100;
    const W: usize = 200;
    const R_BLOCKS: usize = 50;
    const C_BLOCKS: usize = 100;
    const K: usize = R_BLOCKS * C_BLOCKS; // 5000

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for _ in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for _ in 0..W { row.push(sp1_zkvm::io::read::<f64>()); }
        a.push(row);
    }
    // Claimed result: K * 2 * 2 floats.
    let mut claimed: Vec<f64> = Vec::with_capacity(K * 4);
    for _ in 0..(K * 4) { claimed.push(sp1_zkvm::io::read::<f64>()); }

    // Compute result with the same kernel order:
    // for c_block in 0..C_BLOCKS: for r_block in 0..R_BLOCKS:
    //   for p in 0..2: for q in 0..2: result[k][p][q] = a[r_block*2+p][c_block*2+q]
    let mut result: Vec<f64> = Vec::with_capacity(K * 4);
    for c_block in 0..C_BLOCKS {
        for r_block in 0..R_BLOCKS {
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
