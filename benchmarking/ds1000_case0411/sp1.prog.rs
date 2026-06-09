// DS-1000 case 0411 — SP1 guest program.
// Kernel: result = fill_crop(a, [low_index, low_index], np.zeros((H-L, H-L)))
// fill_crop semantics: result[i,j] = a[low_index+i, low_index+j] when in bounds
// else 0. Output shape: (H-L, H-L). a: Float[5,5].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const R: i64 = 5;
    let low: i64 = sp1_zkvm::io::read::<i64>();
    let high: i64 = sp1_zkvm::io::read::<i64>();
    let mut a = [[0.0f64; 5]; 5];
    for i in 0..5usize { for j in 0..5usize { a[i][j] = sp1_zkvm::io::read::<f64>(); } }
    let h = (high - low) as usize;
    let mut out = vec![vec![0.0f64; h]; h];
    for i in 0..h {
        for j in 0..h {
            let si = low + i as i64;
            let sj = low + j as i64;
            if si >= 0 && si < R && sj >= 0 && sj < R {
                out[i][j] = a[si as usize][sj as usize];
            }
        }
    }
    for row in &out {
        for v in row {
            let claim: f64 = sp1_zkvm::io::read::<f64>();
            assert!((v - claim).abs() < 1e-9);
        }
    }
}
