// DS-1000 case 0362 — SP1 guest program.
// Kernel: mask = del_col <= 6; del_col = del_col[mask] - 1; result = np.delete(a, del_col, axis=1).
// a: Float[8,6], del_col: Integer[4].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const R: usize = 8;
    const C: usize = 6;
    const D: usize = 4;
    let out_cols: u32 = sp1_zkvm::io::read::<u32>();
    let mut a = [[0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = sp1_zkvm::io::read::<f64>(); } }
    let mut del_col = [0i64; D];
    for k in 0..D { del_col[k] = sp1_zkvm::io::read::<i64>(); }
    let mut delete_mask = [false; C];
    for k in 0..D {
        let v = del_col[k];
        if v <= C as i64 {
            let idx = (v - 1) as usize;
            if idx < C { delete_mask[idx] = true; }
        }
    }
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(R);
    for i in 0..R {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..C { if !delete_mask[j] { row.push(a[i][j]); } }
        out.push(row);
    }
    assert_eq!(out[0].len() as u32, out_cols);
    for row in &out {
        for v in row {
            let claim: f64 = sp1_zkvm::io::read::<f64>();
            assert!((v - claim).abs() < 1e-9);
        }
    }
}
