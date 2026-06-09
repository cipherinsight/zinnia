// DS-1000 case 0439 -- Noir baseline.
// X.T[:, :, None] * X.T[:, None] on Float[10,20] -> shape (20, 10, 10).
// X is (10, 20); X.T[r, i] = X[i, r]. Output o[r, i, j] = X[i, r] * X[j, r].

use noir_fpc::qmul;

global ROWS: u32 = 10;
global COLS: u32 = 20;

fn main(x: [Field; 200]) -> pub [Field; 2000] {
    let mut out: [Field; 2000] = [0; 2000];
    for r in 0..COLS {
        for i in 0..ROWS {
            for j in 0..ROWS {
                out[r * ROWS * ROWS + i * ROWS + j] = qmul(x[i * COLS + r], x[j * COLS + r]);
            }
        }
    }
    out
}
