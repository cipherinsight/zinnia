// DS-1000 case 0462, Noir baseline.
// Kernel: `selection[1:] = a[1:] != a[:-1]` (first element selected),
// `selection &= a != 0`, `a[selection]` on Integer[20]: drop zeros and
// repeats of the previous value (data-dependent length).
// Output: the count of kept values, then the kept values in order,
// zero-padded to 20. Only equalities decide the selection (no range check);
// compaction by noir_hw::select_kept (a running one-hot).
// @outputs: dynamic int[20x1]

use noir_hw::select_kept;

global N: u32 = 20;

fn main(a: [Field; N]) -> pub [Field; N + 1] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut keep: [bool; N] = [false; N];
    keep[0] = a[0] != 0;
    for i in 1..N {
        keep[i] = (a[i] != 0) & (a[i] != a[i - 1]);
    }
    select_kept(a, keep)
}

