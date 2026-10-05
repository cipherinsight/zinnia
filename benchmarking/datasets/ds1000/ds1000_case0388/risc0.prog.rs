// DS-1000 case 388 — Risc0 guest program.

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
    let p: u64 = env::read();
    let pu: usize = p as usize;

    let h_eff: usize = (H / pu) * pu;
    let w_eff: usize = (W / pu) * pu;
    let r_blocks: usize = h_eff / pu;
    let c_blocks: usize = w_eff / pu;
    let out_len: usize = r_blocks * c_blocks * pu * pu;

    let mut claimed: Vec<f64> = Vec::with_capacity(out_len);
    for _ in 0..out_len { claimed.push(env::read()); }

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
