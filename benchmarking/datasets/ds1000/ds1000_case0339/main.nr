// DS-1000 case 0339 -- Noir baseline.
// dim = min(a.shape) = 8; b = a[:8, :8]; vstack(diag(b), diag(fliplr(b))).
// On Float[8,10]: diag length 8, output (2, 8).

global N: u32 = 8;
global COLS: u32 = 10;

fn main(a: [Field; 80]) -> pub [Field; 16] {
    let mut out: [Field; 16] = [0; 16];
    for i in 0..N {
        out[i] = a[i * COLS + i];
        out[N + i] = a[i * COLS + (N - 1 - i)];
    }
    out
}
