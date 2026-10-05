// DS-1000 case 0463, Noir baseline.
// Kernel: the (20, 1) column variant of 0462: `selection[1:] = a[1:] !=
// a[:-1]` (first row selected), `selection &= a != 0`,
// `a[selection].reshape(-1, 1)` on Integer[20, 1] (data-dependent length).
// Output: the count of kept rows, then the kept values in order, zero-padded
// to 20 rows of one. Only equalities decide the selection (no range check);
// compaction by noir_hw::select_kept (a running one-hot).
// @outputs: dynamic int[20x1]

use noir_hw::select_kept;

global N: u32 = 20;

fn main(a: [Field; N]) -> pub [Field; N + 1] {
    let mut keep: [bool; N] = [false; N];
    keep[0] = a[0] != 0;
    for i in 1..N {
        keep[i] = (a[i] != 0) & (a[i] != a[i - 1]);
    }
    select_kept(a, keep)
}

