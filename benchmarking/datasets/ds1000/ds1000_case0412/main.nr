// DS-1000 case 0412, Noir baseline.
// Kernel: `x[x >= 0]` on Float[10] (data-dependent length).
// Output: the count of kept values, then the kept values in order, zero-padded
// to 10; compaction writes each kept value at the running count (native
// dynamic array writes).
// @outputs: dynamic float[10x1]

use noir_hw::is_neg;

fn main(x: [Field; 10]) -> pub [Field; 11] {
    let mut kept: [Field; 10] = [0; 10];
    let mut count: u32 = 0;
    for i in 0..10 {
        if !is_neg(x[i]) {
            kept[count] = x[i];
            count += 1;
        }
    }
    let mut out: [Field; 11] = [0; 11];
    out[0] = count as Field;
    for j in 0..10 {
        out[j + 1] = kept[j];
    }
    out
}
