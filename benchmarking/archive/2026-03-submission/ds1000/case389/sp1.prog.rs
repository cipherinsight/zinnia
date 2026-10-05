// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read a (4x2x3)
    let mut a: Vec<Vec<Vec<i32>>> = Vec::new();
    for _ in 0..4 {
        let mut mat: Vec<Vec<i32>> = Vec::new();
        for _ in 0..2 {
            let mut row: Vec<i32> = Vec::new();
            for _ in 0..3 {
                row.push(sp1_zkvm::io::read::<i32>());
            }
            mat.push(row);
        }
        a.push(mat);
    }

    // read result (4x6)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..4 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..6 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        result.push(row);
    }

    let nrows: usize = 2;
    let ncols: usize = 3;
    let h: usize = 4;
    let w: usize = 6;

    // step1: reshape (4,2,3) -> (2,2,2,3)
    let mut flat_a: Vec<i32> = Vec::new();
    for i in 0..4 {
        for r in 0..nrows {
            for c in 0..ncols {
                flat_a.push(a[i][r][c]);
            }
        }
    }
    let mut step1: Vec<Vec<Vec<Vec<i32>>>> = vec![vec![vec![vec![0; ncols]; nrows]; 2]; h / nrows];
    {
        let mut idx = 0usize;
        for i0 in 0..(h / nrows) {
            for i1 in 0..2 {
                for i2 in 0..nrows {
                    for i3 in 0..ncols {
                        step1[i0][i1][i2][i3] = flat_a[idx];
                        idx += 1;
                    }
                }
            }
        }
    }

    // step2: transpose (0,2,1,3)
    let mut step2: Vec<Vec<Vec<Vec<i32>>>> = vec![vec![vec![vec![0; ncols]; 2]; 2]; h / nrows];
    for i0 in 0..(h / nrows) {
        for i1 in 0..2 {
            for i2 in 0..2 {
                for i3 in 0..ncols {
                    step2[i0][i2][i1][i3] = step1[i0][i1][i2][i3];
                }
            }
        }
    }

    // computed: reshape (2,2,2,3) -> (4,6)
    let mut flat_step2: Vec<i32> = Vec::new();
    for i0 in 0..(h / nrows) {
        for i1 in 0..2 {
            for i2 in 0..2 {
                for i3 in 0..ncols {
                    flat_step2.push(step2[i0][i1][i2][i3]);
                }
            }
        }
    }
    let mut computed: Vec<Vec<i32>> = vec![vec![0; w]; h];
    {
        let mut idx = 0usize;
        for r in 0..h {
            for c in 0..w {
                computed[r][c] = flat_step2[idx];
                idx += 1;
            }
        }
    }

    // Compare
    for r in 0..h {
        for c in 0..w {
            assert_eq!(result[r as usize][c as usize], computed[r as usize][c as usize]);
        }
    }

    // sp1_zkvm::io::commit_slice(&output);
}
