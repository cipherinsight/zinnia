// DS-1000 case 0444, Noir baseline.
// Kernel: `A[(A > B[0]) & (A < B[1]) | (A > B[1]) & (A < B[2])]` on
// Integer[20], Integer[3] (data-dependent length).
// Output: the count of kept values, then the kept values in order,
// zero-padded to 20. Each value is windowed once; compaction by
// noir_hw::select_kept (a running one-hot).
// @outputs: dynamic int[20x1]

use noir_hw::{select_kept, window};

global N: u32 = 20;

fn main(A: [Field; N], B: [Field; 3]) -> pub [Field; N + 1] {
    let b = [window(B[0]), window(B[1]), window(B[2])];
    let mut keep: [bool; N] = [false; N];
    for i in 0..N {
        let w = window(A[i]);
        keep[i] = ((w > b[0]) & (w < b[1])) | ((w > b[1]) & (w < b[2]));
    }
    select_kept(A, keep)
}

