// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure (reshape -> transpose -> reshape) are faithfully preserved.

#[executable]
pub fn main() {
    // a: NDArray[int, 4, 2, 3]
    // Blocks laid out in a single dimension, each of shape 2x3.
    let a = array![
        array![array![0_u32, 1_u32, 2_u32],  array![6_u32, 7_u32, 8_u32]],
        array![array![3_u32, 4_u32, 5_u32],  array![9_u32, 10_u32, 11_u32]],
        array![array![12_u32, 13_u32, 14_u32], array![18_u32, 19_u32, 20_u32]],
        array![array![15_u32, 16_u32, 17_u32], array![21_u32, 22_u32, 23_u32]],
    ];

    // result: NDArray[int, 4, 6]
    let result = array![
        array![0_u32, 1_u32, 2_u32, 3_u32, 4_u32, 5_u32],
        array![6_u32, 7_u32, 8_u32, 9_u32, 10_u32, 11_u32],
        array![12_u32, 13_u32, 14_u32, 15_u32, 16_u32, 17_u32],
        array![18_u32, 19_u32, 20_u32, 21_u32, 22_u32, 23_u32],
    ];

    // Constants from the Zinnia code
    let nrows: u32 = 2_u32;
    let ncols: u32 = 3_u32;
    let h: u32 = 4_u32;
    let w: u32 = 6_u32;

    // step1 = a.reshape((h // nrows, 2, nrows, ncols))   // (2, 2, 2, 3)
    // Map (R, C, r, c) -> a[p, r, c] with p = R*2 + C
    let mut step1 = ArrayTrait::new(); // [2][2][2][3]
    for R in 0..(h / nrows) { // 0..2
        let mut lvl1 = ArrayTrait::new();
        for C in 0..2_u32 {   // 0..2
            let mut lvl2 = ArrayTrait::new();
            let p: u32 = R * 2_u32 + C;
            for r in 0..nrows {
                let mut lvl3 = ArrayTrait::new();
                for c in 0..ncols {
                    lvl3.append(*a.at(p).at(r).at(c));
                }
                lvl2.append(lvl3);
            }
            lvl1.append(lvl2);
        }
        step1.append(lvl1);
    }

    // step2 = step1.transpose((0, 2, 1, 3))  // swapaxes(1,2)
    // step2[R][r][C][c] = step1[R][C][r][c]
    let mut step2 = ArrayTrait::new(); // [2][2][2][3]
    for R in 0..(h / nrows) {
        let mut x0 = ArrayTrait::new();
        for r in 0..nrows {
            let mut x1 = ArrayTrait::new();
            for C in 0..2_u32 {
                let mut x2 = ArrayTrait::new();
                for c in 0..ncols {
                    x2.append(*step1.at(R).at(C).at(r).at(c));
                }
                x1.append(x2);
            }
            x0.append(x1);
        }
        step2.append(x0);
    }

    // computed = step2.reshape((h, w)) // (4, 6)
    // Row index = R*nrows + r; Col index = C*ncols + c
    let mut computed = ArrayTrait::new(); // [4][6]
    for row in 0..h {
        let mut out_row = ArrayTrait::new();
        let R = row / nrows;
        let r = row % nrows;
        for col in 0..w {
            let C = col / ncols;
            let c = col % ncols;
            out_row.append(*step2.at(R).at(r).at(C).at(c));
        }
        computed.append(out_row);
    }

    // Assert equality with provided result
    for i in 0..h {
        for j in 0..w {
            assert!(*result.at(i).at(j) == *computed.at(i).at(j));
        }
    }
}
