// DS-1000 case 368 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [[i64; 3]; 3] = [[0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = env::read();
        }
    }
    let claimed: u8 = env::read();

    let rtol: f64 = 1e-5;
    let mut rtol_abs_b: [f64; 3] = [0.0; 3];
    for j in 0..3 {
        let b: f64 = a[0][j] as f64;
        let abs_b: f64 = if b < 0.0 { -b } else { b };
        rtol_abs_b[j] = abs_b * rtol;
    }

    let mut all_close: bool = true;
    for i in 0..3 {
        for j in 0..3 {
            let diff: i64 = a[i][j] - a[0][j];
            let diff_fp: f64 = diff as f64;
            let abs_d: f64 = if diff_fp < 0.0 { -diff_fp } else { diff_fp };
            let leq: bool = abs_d <= rtol_abs_b[j];
            all_close = all_close && leq;
        }
    }

    let expected: bool = claimed != 0;
    assert_eq!(all_close, expected);
}
