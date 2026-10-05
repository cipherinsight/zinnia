// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a: NDArray[int, 5, 6]
    let a = array![
        array![0_u32, 1_u32, 2_u32, 3_u32, 4_u32, 5_u32],
        array![5_u32, 6_u32, 7_u32, 8_u32, 9_u32, 10_u32],
        array![10_u32, 11_u32, 12_u32, 13_u32, 14_u32, 15_u32],
        array![15_u32, 16_u32, 17_u32, 18_u32, 19_u32, 20_u32],
        array![20_u32, 21_u32, 22_u32, 23_u32, 24_u32, 25_u32],
    ];

    // result: NDArray[int, 2, 5]
    let result = array![
        array![0_u32, 6_u32, 12_u32, 18_u32, 24_u32],
        array![4_u32, 8_u32, 12_u32, 16_u32, 20_u32],
    ];

    // nrows = 5, ncols = 6, dim = min(nrows, ncols) = 5
    let dim: u32 = 5_u32;

    // b = a[:dim, :dim]  (extract 5x5 leading submatrix)
    let mut b = ArrayTrait::new();
    for i in 0..dim {
        let mut row = ArrayTrait::new();
        for j in 0..dim {
            row.append(*a.at(i).at(j));
        }
        b.append(row);
    }

    // main_diag[i] = b[i, i]
    let mut main_diag = ArrayTrait::new();
    for i in 0..dim {
        main_diag.append(*b.at(i).at(i));
    }

    // flipped[i, j] = b[i, (dim - 1) - j]  // fliplr
    let mut flipped = ArrayTrait::new();
    for i in 0..dim {
        let mut row = ArrayTrait::new();
        for j in 0..dim {
            row.append(*b.at(i).at((dim - 1_u32) - j));
        }
        flipped.append(row);
    }

    // anti_diag[i] = flipped[i, i]
    let mut anti_diag = ArrayTrait::new();
    for i in 0..dim {
        anti_diag.append(*flipped.at(i).at(i));
    }

    // stacked = vstack(main_diag, anti_diag) -> shape (2, dim)
    let mut stacked = ArrayTrait::new();
    let mut row0 = ArrayTrait::new();
    let mut row1 = ArrayTrait::new();
    for i in 0..dim {
        row0.append(*main_diag.at(i));
        row1.append(*anti_diag.at(i));
    }
    stacked.append(row0);
    stacked.append(row1);

    // assert result == stacked
    for r in 0..2_u32 {
        for c in 0..dim {
            assert!(*result.at(r).at(c) == *stacked.at(r).at(c));
        }
    }
}
