// DS-1000 case 0412, Noir baseline.
// Kernel: `x[x >= 0]` on Float[10] (data-dependent length).
// Output: the count of kept values, then the kept values in order, zero-padded
// to 10 (compaction: noir_hw::compact).
// @outputs: dynamic float[10x1]

use noir_hw::{compact, is_neg};

fn main(x: [Field; 10]) -> pub [Field; 11] {
    let mut keep: [Field; 10] = [0; 10];
    let mut w: [Field; 10] = [0; 10];
    let mut count: Field = 0;
    for i in 0..10 {
        keep[i] = (!is_neg(x[i])) as Field;
        w[i] = keep[i] * x[i];
        count += keep[i];
    }
    let kept = compact(keep, w);
    let mut out: [Field; 11] = [0; 11];
    out[0] = count;
    for j in 0..10 {
        out[j + 1] = kept[j];
    }
    out
}
