// DS-1000 case 319 — SP1 guest program.
// Kernel:
//   c = empty_like(permutation)
//   c[permutation] = arange(len(permutation))   # inverse permutation
//   result = a[c, :, :]                          # reorder rows by c
// Inputs: a: Float[8, 6, 8], permutation: Integer[8].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<Vec<Vec<f64>>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut plane: Vec<Vec<f64>> = Vec::with_capacity(6);
        for _ in 0..6 {
            let mut row: Vec<f64> = Vec::with_capacity(8);
            for _ in 0..8 {
                row.push(sp1_zkvm::io::read::<f64>());
            }
            plane.push(row);
        }
        a.push(plane);
    }
    let mut permutation: [i64; 8] = [0; 8];
    for k in 0..8 {
        permutation[k] = sp1_zkvm::io::read::<i64>();
    }

    // Build inverse permutation c.
    let mut c: [i64; 8] = [0; 8];
    for k in 0..8 {
        let p: usize = permutation[k] as usize;
        c[p] = k as i64;
    }

    // result[i][j][k] = a[c[i]][j][k].
    for i in 0..8 {
        let src_i: usize = c[i] as usize;
        for j in 0..6 {
            for k in 0..8 {
                let claim_v: f64 = sp1_zkvm::io::read::<f64>();
                assert!((a[src_i][j][k] - claim_v).abs() < 1e-12);
            }
        }
    }
}
