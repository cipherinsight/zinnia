// DS-1000 case 0435 -- Noir baseline.
// a[1, :] = 0; a[:, 0] = 0 on Integer[4,4]. Comptime indices.

global ROWS: u32 = 4;
global COLS: u32 = 4;

fn main(a: [Field; 16]) -> pub [Field; 16] {
    let mut out: [Field; 16] = a;
    for j in 0..COLS { out[1 * COLS + j] = 0; }
    for i in 0..ROWS { out[i * COLS + 0] = 0; }
    out
}
