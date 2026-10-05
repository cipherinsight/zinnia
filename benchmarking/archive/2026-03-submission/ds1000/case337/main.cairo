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

    // result: NDArray[int, 5]
    let result = array![5_u32, 9_u32, 13_u32, 17_u32, 21_u32];

    // nrows = a.shape[0], ncols = a.shape[1]
    let nrows: u32 = 5_u32;
    let ncols: u32 = 6_u32;

    // flipped[i, j] = a[i, (ncols - 1) - j]  // fliplr
    let mut flipped = ArrayTrait::new();
    for i in 0..nrows {
        let mut row = ArrayTrait::new();
        for j in 0..ncols {
            let val = *a.at(i).at((ncols - 1_u32) - j);
            row.append(val);
        }
        flipped.append(row);
    }

    // diag_vals[k] = flipped[k, k]
    let mut diag_vals = ArrayTrait::new();
    for k in 0..nrows {
        diag_vals.append(*flipped.at(k).at(k));
    }

    // assert result == diag_vals
    for i in 0..nrows {
        assert!(*result.at(i) == *diag_vals.at(i));
    }
}
