// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read a (4x4)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..4 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..4 {
            row.push(sp1_zkvm::io::read::<i32>());
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
                r.push(sp1_zkvm::io::read::<i32>());
            }
            m.push(r);
        }
        result.push(m);
    }

    // Step 1: reshape a -> (2,2,2,2) via flat row-major mapping
    let mut flat: Vec<i32> = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            flat.push(a[i as usize][j as usize]);
        }
    }
    let mut reshaped = vec![vec![vec![vec![0i32; 2]; 2]; 2]; 2];
    for i0 in 0..2 {
        for i1 in 0..2 {
            for i2 in 0..2 {
                for i3 in 0..2 {
                    let idx: usize = (((i0 * 2 + i1) * 2 + i2) * 2 + i3) as usize;
                    reshaped[i0 as usize][i1 as usize][i2 as usize][i3 as usize] = flat[idx];
                }
            }
        }
    }

    // Step 2: transpose (0,2,1,3) then (1,0,2,3)
    let mut tmp1 = vec![vec![vec![vec![0i32; 2]; 2]; 2]; 2];
    for a0 in 0..2 {
        for a1 in 0..2 {
            for a2 in 0..2 {
                for a3 in 0..2 {
                    tmp1[a0 as usize][a1 as usize][a2 as usize][a3 as usize] =
                        reshaped[a0 as usize][a2 as usize][a1 as usize][a3 as usize];
                }
            }
        }
    }
    let mut tmp2 = vec![vec![vec![vec![0i32; 2]; 2]; 2]; 2];
    for a0 in 0..2 {
        for a1 in 0..2 {
            for a2 in 0..2 {
                for a3 in 0..2 {
                    tmp2[a0 as usize][a1 as usize][a2 as usize][a3 as usize] =
                        tmp1[a1 as usize][a0 as usize][a2 as usize][a3 as usize];
                }
            }
        }
    }

    // Step 3: reshape (2,2,2,2) -> (4,2,2)
    let mut computed: Vec<Vec<Vec<i32>>> = vec![vec![vec![0i32; 2]; 2]; 4];
    for u in 0..2 {
        for v in 0..2 {
            let p: usize = (u * 2 + v) as usize;
            for w in 0..2 {
                for x in 0..2 {
                    computed[p][w as usize][x as usize] =
                        tmp2[u as usize][v as usize][w as usize][x as usize];
                }
            }
        }
    }

    // Compare result == computed
    for p in 0..4 {
        for w in 0..2 {
            for x in 0..2 {
                assert_eq!(result[p as usize][w as usize][x as usize], computed[p as usize][w as usize][x as usize]);
            }
        }
    }

    // sp1_zkvm::io::commit_slice(&bytes);
}
