// DS-1000 case 0422, Noir baseline.
// Kernel: `np.correlate(a, np.hstack((b[1:], b)), mode='valid')` on two Float[50].
// v = hstack((b[1:], b)) has 99 elements, v[m] = b[(m + 1) % 50]. NumPy's
// 'valid' correlation of the shorter a with v is
// out[k] = sum_n a[n] * v[n + 49 - k] = sum_n a[n] * b[(n - k) mod 50], k < 50:
// a circular correlation, each output one dot product with one rescale.
// @outputs: float[50]

use noir_hw::rescale;

fn main(a: [Field; 50], b: [Field; 50]) -> pub [Field; 50] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    let mut corr = [0; 50];
    for k in 0..50 {
        let mut acc: Field = 0;
        for n in 0..50 {
            acc += a[n] * b[(n + 50 - k) % 50];
        }
        corr[k] = rescale(acc);
    }
    corr
}
