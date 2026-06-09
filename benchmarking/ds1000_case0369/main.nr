// DS-1000 case 0369 -- Noir baseline.
// np.isclose(a, a[:, 0].reshape(-1, 1), atol=0).all() on Integer[3,3].
// Each row's column-0 broadcast as comparison reference.

global ROWS: u32 = 3;
global COLS: u32 = 3;

fn main(a: [Field; 9]) -> pub Field {
    let mut all_equal: Field = 1;
    for i in 0..ROWS {
        for j in 0..COLS {
            let eq: Field = if a[i * COLS + j] == a[i * COLS] { 1 } else { 0 };
            all_equal = all_equal * eq;
        }
    }
    all_equal
}
