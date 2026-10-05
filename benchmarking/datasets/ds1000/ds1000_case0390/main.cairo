// DS-1000 case 390 — Cairo baseline. Patch reshape over Float[100,200].
// Kernel:
//   x = a[:H//ps*ps, :W//ps*ps]
//   result = x.reshape(H//ps, ps, W//ps, ps).swapaxes(1,2).transpose(1,0,2,3)
//             .reshape(-1, ps, ps)
// Sample: a[i][j] = i*200 + j; ps = 10. Patch p has block_col=p/10,
// block_row=p%10; patch[r][c] = a[block_row*10+r][block_col*10+c].
// Quantisation ×2^32 to match Zinnia.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let h: u32 = 100;
    let w: u32 = 200;
    let ps: u32 = 10;
    let h_blocks: u32 = h / ps;
    let w_blocks: u32 = w / ps;
    let n_patches: u32 = h_blocks * w_blocks;

    // Build a flat Q-scaled array.
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..h {
        for j in 0_u32..w {
            let v: u128 = (i * 200 + j).into();
            a.append(v * Q);
        };
    };

    // Trim. (Here H,W already multiples of ps, but mirror the kernel.)
    let trimmed_h: u32 = h_blocks * ps;
    let trimmed_w: u32 = w_blocks * ps;
    let mut x: Array<u128> = ArrayTrait::new();
    for i in 0_u32..trimmed_h {
        for j in 0_u32..trimmed_w {
            let idx: u32 = i * w + j;
            x.append(*a.at(idx.try_into().unwrap()));
        };
    };

    for p in 0_u32..n_patches {
        let block_col: u32 = p / h_blocks;
        let block_row: u32 = p % h_blocks;
        for r in 0_u32..ps {
            for c in 0_u32..ps {
                let row: u32 = block_row * ps + r;
                let col: u32 = block_col * ps + c;
                let xi: u32 = row * trimmed_w + col;
                let v: u128 = *x.at(xi.try_into().unwrap());
                let expected_int: u128 = (row * 200 + col).into();
                let expected: u128 = expected_int * Q;
                assert!(v == expected);
            };
        };
    };
}
