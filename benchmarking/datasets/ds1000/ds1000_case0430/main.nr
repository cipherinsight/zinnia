// DS-1000 case 0430, Noir baseline.
// Kernel: `(a.mean() - 3*a.std(), a.mean() + 3*a.std())` on Float[30].
// a.std() appears twice in the kernel and is computed once; the squared
// deviations are summed with one rescale.
// @outputs: float[2]

use noir_hw::{floor_div_const, rescale, sqrt};

fn main(a: [Field; 30]) -> pub [Field; 2] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut total: Field = 0;
    for i in 0..30 {
        total += a[i];
    }
    let mean = floor_div_const::<30>(total);
    let mut sq_sum: Field = 0;
    for i in 0..30 {
        let d = a[i] - mean;
        sq_sum += d * d;
    }
    let var = floor_div_const::<30>(rescale(sq_sum));
    let std = sqrt(var);
    [mean - 3 * std, mean + 3 * std]
}
