// DS-1000 case 390 — SP1 guest program.
// Kernel:
//   x = a[:H//ps*ps, :W//ps*ps]
//   result = x.reshape(H//ps, ps, W//ps, ps)
//             .swapaxes(1, 2).transpose(1, 0, 2, 3)
//             .reshape(-1, ps, ps)
// Inputs: a Float[100, 200], patch_size scalar Integer; expected result is
// Float[N_patches, ps, ps] where N_patches = (H//ps)*(W//ps).
//
// Sample uses H=100, W=200, ps=10 → N_patches=200.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const H: usize = 100;
    const W: usize = 200;
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for _ in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for _ in 0..W {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        a.push(row);
    }
    let patch_size: u32 = sp1_zkvm::io::read::<u32>();
    let ps = patch_size as usize;
    let h_blocks = H / ps;
    let w_blocks = W / ps;
    let n_patches = h_blocks * w_blocks;

    let mut claimed: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_patches);
    for _ in 0..n_patches {
        let mut block: Vec<Vec<f64>> = Vec::with_capacity(ps);
        for _ in 0..ps {
            let mut row: Vec<f64> = Vec::with_capacity(ps);
            for _ in 0..ps {
                row.push(sp1_zkvm::io::read::<f64>());
            }
            block.push(row);
        }
        claimed.push(block);
    }

    // Trim.
    let trimmed_h = h_blocks * ps;
    let trimmed_w = w_blocks * ps;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(trimmed_h);
    for i in 0..trimmed_h {
        let mut row: Vec<f64> = Vec::with_capacity(trimmed_w);
        for j in 0..trimmed_w {
            row.push(a[i][j]);
        }
        x.push(row);
    }

    // After reshape(h_blocks, ps, w_blocks, ps).swapaxes(1,2).transpose(1,0,2,3)
    // .reshape(-1, ps, ps): patch p corresponds to (block_col=p/h_blocks,
    // block_row=p%h_blocks); patch[r][c] = x[block_row*ps + r][block_col*ps + c].
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
