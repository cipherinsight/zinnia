// DS-1000 case 0322 -- Noir baseline.
// np.argwhere(a == np.min(a)) on Integer[2,2]. Output (first_row, first_col, length).
// Mirrors halo2 baseline: scan min, count indicators, take first match.

use noir_fpc::qlt;

fn main(a: [Field; 4]) -> pub [Field; 3] {
    let mut min_val: Field = a[0];
    for k in 1..4 {
        if qlt(a[k], min_val) { min_val = a[k]; }
    }
    let mut indicators: [Field; 4] = [0; 4];
    let mut length: Field = 0;
    for k in 0..4 {
        let ind: Field = if a[k] == min_val { 1 } else { 0 };
        indicators[k] = ind;
        length = length + ind;
    }
    let mut first_row: Field = 0;
    let mut first_col: Field = 0;
    let mut found: Field = 0;
    for k in 0..4 {
        let take = (1 - found) * indicators[k];
        first_row = first_row + take * ((k / 2) as Field);
        first_col = first_col + take * ((k % 2) as Field);
        let prod = found * indicators[k];
        found = found + indicators[k] - prod;
    }
    [first_row, first_col, length]
}
