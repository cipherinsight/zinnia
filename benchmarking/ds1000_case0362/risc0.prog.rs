// DS-1000 case 0362 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const R: usize = 8;
    const C: usize = 6;
    const D: usize = 4;
    let out_cols: u32 = env::read();
    let mut a = [[0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = env::read(); } }
    let mut del_col = [0i64; D];
    for k in 0..D { del_col[k] = env::read(); }
    let mut delete_mask = [false; C];
    for k in 0..D {
        let v = del_col[k];
        if v >= 1 && v <= C as i64 { delete_mask[(v - 1) as usize] = true; }
    }
    for i in 0..R {
        let mut count = 0u32;
        for j in 0..C {
            if !delete_mask[j] {
                let claim: f64 = env::read();
                assert!((a[i][j] - claim).abs() < 1e-9);
                count += 1;
            }
        }
        if i == 0 { assert_eq!(count, out_cols); }
    }
}
