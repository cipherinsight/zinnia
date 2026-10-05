// DS-1000 case 0343 -- Noir baseline. X.T.flat on Integer[5,6].
// Transposed flatten: out[j*5 + i] = X[i*6 + j].

global ROWS: u32 = 5;
global COLS: u32 = 6;

fn main(x: [Field; 30]) -> pub [Field; 30] {
    let mut out: [Field; 30] = [0; 30];
    let mut p: u32 = 0;
    for j in 0..COLS {
        for i in 0..ROWS {
            out[p] = x[i * COLS + j];
            p = p + 1;
        }
    }
    out
}
