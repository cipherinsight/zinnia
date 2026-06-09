// DS-1000 case 388 — SP1 guest program.
// Kernel: x = a[:H/p*p, :W/p*p]; x.reshape(H_eff/p, p, W_eff/p, p).swapaxes(1, 2).reshape(-1, p, p)
// With H=100, W=200, p=2: H_eff=100, W_eff=200, R_blocks=50, C_blocks=100, K=5000.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const H: usize = 100;
    const W: usize = 200;

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for _ in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for _ in 0..W { row.push(sp1_zkvm::io::read::<f64>()); }
        a.push(row);
    }
    let p: u64 = sp1_zkvm::io::read::<u64>();
    let pu: usize = p as usize;

    let h_eff: usize = (H / pu) * pu;
    let w_eff: usize = (W / pu) * pu;
    let r_blocks: usize = h_eff / pu;
    let c_blocks: usize = w_eff / pu;
    let k_total: usize = r_blocks * c_blocks;
    let out_len: usize = k_total * pu * pu;

    let mut claimed: Vec<f64> = Vec::with_capacity(out_len);
    for _ in 0..out_len { claimed.push(sp1_zkvm::io::read::<f64>()); }

    // x = a[:h_eff, :w_eff]: implicit; we just index.
    let mut result: Vec<f64> = Vec::with_capacity(out_len);
    for r_block in 0..r_blocks {
        for c_block in 0..c_blocks {
            for pp in 0..pu {
                for qq in 0..pu {
                    result.push(a[r_block * pu + pp][c_block * pu + qq]);
                }
            }
        }
    }

    for i in 0..out_len {
        assert!((result[i] - claimed[i]).abs() < 1e-9);
    }
}
