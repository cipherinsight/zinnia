// DS-1000 case 390 — Risc0 guest. Patch reshape over Float[100,200].

use risc0_zkvm::guest::env;

fn main() {
    const H: usize = 100;
    const W: usize = 200;
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for _ in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for _ in 0..W { row.push(env::read()); }
        a.push(row);
    }
    let patch_size: u32 = env::read();
    let ps = patch_size as usize;
    let h_blocks = H / ps;
    let w_blocks = W / ps;
    let n_patches = h_blocks * w_blocks;

    let mut claimed: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_patches);
    for _ in 0..n_patches {
        let mut block: Vec<Vec<f64>> = Vec::with_capacity(ps);
        for _ in 0..ps {
            let mut row: Vec<f64> = Vec::with_capacity(ps);
            for _ in 0..ps { row.push(env::read()); }
            block.push(row);
        }
        claimed.push(block);
    }

    let trimmed_h = h_blocks * ps;
    let trimmed_w = w_blocks * ps;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(trimmed_h);
    for i in 0..trimmed_h {
        let mut row: Vec<f64> = Vec::with_capacity(trimmed_w);
        for j in 0..trimmed_w { row.push(a[i][j]); }
        x.push(row);
    }

    for p in 0..n_patches {
        let block_col = p / h_blocks;
        let block_row = p % h_blocks;
        for r in 0..ps {
            for c in 0..ps {
                let v = x[block_row * ps + r][block_col * ps + c];
                assert!((v - claimed[p][r][c]).abs() < 0.001);
            }
        }
    }
}
