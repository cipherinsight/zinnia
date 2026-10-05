// DS-1000 case 0444, Noir baseline.
// Kernel: `A[(A > B[0]) & (A < B[1]) | (A > B[1]) & (A < B[2])]` on
// Integer[20], Integer[3] (data-dependent length).
// Output: the count of kept values, then the kept values in order,
// zero-padded to 20. Each value is windowed once; compaction by a running
// one-hot (`compact`).
// @outputs: dynamic int[20x1]

use noir_hw::window;

global N: u32 = 20;

fn main(A: [Field; N], B: [Field; 3]) -> pub [Field; N + 1] {
    let b = [window(B[0]), window(B[1]), window(B[2])];
    let mut keep: [bool; N] = [false; N];
    for i in 0..N {
        let w = window(A[i]);
        keep[i] = ((w > b[0]) & (w < b[1])) | ((w > b[1]) & (w < b[2]));
    }
    compact(A, keep)
}

// x[keep] at the front of N slots, zero-padded, after the count:
// at[j] = [j values kept before element i] (a running one-hot that moves up
// one place after each kept element); element i adds keep_i x_i to slot at.
// No range checks and no dynamic writes (a conditional native write costs
// several memory operations each).
fn compact(x: [Field; N], keep: [bool; N]) -> [Field; N + 1] {
    let mut out: [Field; N + 1] = [0; N + 1];
    let mut at: [Field; N] = [0; N];
    at[0] = 1;
    let mut count: Field = 0;
    for i in 0..N {
        let k = keep[i] as Field;
        let y = k * x[i];
        let mut next: [Field; N] = [0; N];
        for j in 0..N {
            if j <= i {
                out[j + 1] += at[j] * y;
                let moved = k * at[j];
                next[j] += at[j] - moved;
                if j + 1 < N {
                    next[j + 1] += moved;
                }
            }
        }
        at = next;
        count += k;
    }
    out[0] = count;
    out
}
