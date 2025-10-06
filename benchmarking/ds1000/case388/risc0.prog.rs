// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read a (4x5)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..4 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..5 {
            row.push(env::read::<i32>());
        }
        a.push(row);
    }

    // read result (4x2x2)
    let mut result: Vec<Vec<Vec<i32>>> = Vec::new();
    for _ in 0..4 {
        let mut m: Vec<Vec<i32>> = Vec::new();
        for _ in 0..2 {
            let mut r: Vec<i32> = Vec::new();
            for _ in 0..2 {
                r.push(env::read::<i32>());
            }
            m.push(r);
        }
        result.push(m);
    }

    // Parameters
    let patch: usize = 2;

    // 1) Trim to multiples of patch size
    let rows: usize = (a.len() / patch) * patch;           // 4
    let cols: usize = (a[0].len() / patch) * patch;         // 4
    // Build x = a[:rows, :cols] (4x4)
    let mut x: Vec<Vec<i32>> = Vec::new();
    for i in 0..rows {
        let mut row: Vec<i32> = Vec::new();
        for j in 0..cols {
            row.push(a[i as usize][j as usize]);
        }
        x.push(row);
    }

    // 2) Blockify -> shape (rows/2, 2, cols/2, 2) == (2,2,2,2)
    let rb: usize = rows / patch; // row blocks
    let cb: usize = cols / patch; // col blocks

    // Flatten x row-major, then map to 4D
    let mut flat: Vec<i32> = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            flat.push(x[i as usize][j as usize]);
        }
    }
    let mut blk = vec![vec![vec![vec![0i32; patch]; cb]; patch]; rb];
    for i0 in 0..rb {
        for i1 in 0..patch {
            for i2 in 0..cb {
                for i3 in 0..patch {
                    let idx: usize = (((i0 * patch + i1) * cb + i2) * patch + i3) as usize;
                    blk[i0 as usize][i1 as usize][i2 as usize][i3 as usize] = flat[idx];
                }
            }
        }
    }

    // 3) perm = transpose(blk, (0, 2, 1, 3))
    let mut perm = vec![vec![vec![vec![0i32; patch]; patch]; cb]; rb];
    for a0 in 0..rb {
        for a1 in 0..cb {
            for a2 in 0..patch {
                for a3 in 0..patch {
                    perm[a0 as usize][a1 as usize][a2 as usize][a3 as usize] =
                        blk[a0 as usize][a2 as usize][a1 as usize][a3 as usize];
                }
            }
        }
    }

    // 4) computed = perm reshaped to (rb*cb, 2, 2) == (4,2,2)
    let mut computed: Vec<Vec<Vec<i32>>> = vec![vec![vec![0i32; patch]; patch]; rb * cb];
    for u in 0..rb {
        for v in 0..cb {
            let p: usize = (u * cb + v) as usize;
            for w in 0..patch {
                for x2 in 0..patch {
                    computed[p][w as usize][x2 as usize] =
                        perm[u as usize][v as usize][w as usize][x2 as usize];
                }
            }
        }
    }

    // Compare result == computed
    for p in 0..(rb * cb) {
        for w in 0..patch {
            for x2 in 0..patch {
                assert_eq!(
                    result[p as usize][w as usize][x2 as usize],
                    computed[p as usize][w as usize][x2 as usize]
                );
            }
        }
    }

    // env::commit(&input);
}
