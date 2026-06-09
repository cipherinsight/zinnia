// DS-1000 case 0370 -- Noir baseline. Same as 0368.

global ROWS: u32 = 3;
global COLS: u32 = 3;

fn main(a: [Field; 9]) -> pub Field {
    let mut all_equal: Field = 1;
    for i in 0..ROWS {
        for j in 0..COLS {
            let eq: Field = if a[i * COLS + j] == a[j] { 1 } else { 0 };
            all_equal = all_equal * eq;
        }
    }
    all_equal
}
