// DS-1000 case 318 — SP1 guest program.
// Kernel:
//   c = empty_like(permutation)
//   c[permutation] = arange(len(permutation))   # inverse permutation
//   a = a[:, c]                                   # reorder columns by c
// Inputs: a: Float[8, 6], permutation: Integer[6].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            a[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    let mut permutation: [i64; 6] = [0; 6];
    for k in 0..6 {
        permutation[k] = sp1_zkvm::io::read::<i64>();
    }

    // Build inverse permutation c.
    let mut c: [i64; 6] = [0; 6];
    for k in 0..6 {
        let p: usize = permutation[k] as usize;
        c[p] = k as i64;
    }

    // out[i][j] = a[i][c[j]].
    let mut out: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            let src_col: usize = c[j] as usize;
            out[i][j] = a[i][src_col];
        }
    }

    for i in 0..8 {
        for j in 0..6 {
            let claim_v: f64 = sp1_zkvm::io::read::<f64>();
            assert!((out[i][j] - claim_v).abs() < 1e-12);
        }
    }
}
