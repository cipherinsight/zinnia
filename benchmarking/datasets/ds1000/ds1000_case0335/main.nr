// DS-1000 case 0335, Noir baseline.
// Kernel: `np.max([a, b, c], axis=0)` on three Float[50]: the elementwise maximum.
// Each maximum is noir_hw::max_of: a hint pinned by three range checks and a
// product (cheaper than windowing the three values and comparing twice).
// @outputs: float[50]

use noir_hw::max_of;

fn main(a: [Field; 50], b: [Field; 50], c: [Field; 50]) -> pub [Field; 50] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    noir_hw::check_inputs(c);
    let mut max = [0; 50];
    for i in 0..50 {
        max[i] = max_of([a[i], b[i], c[i]]);
    }
    max
}
