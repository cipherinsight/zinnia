// DS-1000 case 370 — SP1 guest program.
// Kernel: np.isclose(a, a[0], atol=0, rtol=1e-5).all() on Integer[3, 3].
// |a[i][j] - a[0][j]| <= rtol * |a[0][j]| across all 9 cells.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [[i64; 3]; 3] = [[0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = sp1_zkvm::io::read::<i64>();
        }
    }
    let claimed: u8 = sp1_zkvm::io::read::<u8>();

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
