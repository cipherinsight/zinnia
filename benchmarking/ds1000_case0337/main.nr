// DS-1000 case 0337 -- Noir baseline. np.diag(np.fliplr(a)) on Float[8,10].
// fliplr reverses columns; diag picks (i, i). After fliplr, that's a[i][9-i].
// min(rows, cols) determines diag length = 8.

global ROWS: u32 = 8;
global COLS: u32 = 10;

fn main(a: [Field; 80]) -> pub [Field; 8] {
    let mut out: [Field; 8] = [0; 8];
    for i in 0..ROWS {
        out[i] = a[i * COLS + (COLS - 1 - i)];
    }
    out
}
