// DS-1000 case 0431, Noir baseline.
// Kernel: `~((a > mean - 2*std) & (a < mean + 2*std))` on Float[8].
// a.mean() and a.std() appear twice in the kernel and are computed once; the
// squared deviations are summed with one rescale.
// @outputs: bool[8]

use noir_hw::{divmod_pos, rescale, sqrt, window};

fn main(a: [Field; 8]) -> pub [bool; 8] {
    let mut total: Field = 0;
    for i in 0..8 {
        total += a[i];
    }
    let mean = divmod_pos(total, 8).0;
    let mut sq_sum: Field = 0;
    for i in 0..8 {
        let d = a[i] - mean;
        sq_sum += d * d;
    }
    let var = divmod_pos(rescale(sq_sum), 8).0;
    let std = sqrt(var);
    let low = window(mean - 2 * std);
    let high = window(mean + 2 * std);
    let mut outlier = [false; 8];
    for i in 0..8 {
        let x = window(a[i]);
        outlier[i] = !((x > low) & (x < high));
    }
    outlier
}
