// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a =
    // [[ 0,  1,  2,  3,  4],
    //  [ 5,  6,  7,  8,  9],
    //  [10, 11, 12, 13, 14],
    //  [15, 16, 17, 18, 19],
    //  [20, 21, 22, 23, 24]]
    let a = array![
        array![0_u32, 1_u32, 2_u32, 3_u32, 4_u32],
        array![5_u32, 6_u32, 7_u32, 8_u32, 9_u32],
        array![10_u32, 11_u32, 12_u32, 13_u32, 14_u32],
        array![15_u32, 16_u32, 17_u32, 18_u32, 19_u32],
        array![20_u32, 21_u32, 22_u32, 23_u32, 24_u32],
    ];

    // Expected result =
    // [[ 0,  6, 12, 18, 24],
    //  [ 4,  8, 12, 16, 20]]
    let result = array![
        array![0_u32, 6_u32, 12_u32, 18_u32, 24_u32],
        array![4_u32, 8_u32, 12_u32, 16_u32, 20_u32],
    ];

    // n = a.shape[0]
    let n: u32 = 5_u32;

    // main_diag[i] = a[i, i]
    let mut main_diag = ArrayTrait::new();
    for i in 0..n {
        main_diag.append(*a.at(i).at(i));
    }

    // flipped[i, j] = a[i, (n - 1) - j]   // fliplr
    let mut flipped = ArrayTrait::new();
    for i in 0..n {
        let mut row = ArrayTrait::new();
        for j in 0..n {
            row.append(*a.at(i).at((n - 1_u32) - j));
        }
        flipped.append(row);
    }

    // anti_diag[i] = flipped[i, i]
    let mut anti_diag = ArrayTrait::new();
    for i in 0..n {
        anti_diag.append(*flipped.at(i).at(i));
    }

    // stacked = vstack(main_diag, anti_diag) -> shape (2, n)
    let mut stacked = ArrayTrait::new();
    let mut row0 = ArrayTrait::new();
    let mut row1 = ArrayTrait::new();
    for i in 0..n {
        row0.append(*main_diag.at(i));
        row1.append(*anti_diag.at(i));
    }
    stacked.append(row0);
    stacked.append(row1);

    // assert result == stacked
    for r in 0..2_u32 {
        for c in 0..n {
            assert!(*result.at(r).at(c) == *stacked.at(r).at(c));
        }
    }
}
