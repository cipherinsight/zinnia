// DS-1000 case 0491, Noir baseline.
// Kernel: `(~np.isclose(s1, s2)).sum()` on two Float[6, 1]. isclose is NumPy's
// |a - b| <= atol + rtol * |b| with the defaults rtol = 1e-5, atol = 1e-8.
// @outputs: int[1]

use noir_hw::{abs, gt, mul};

global RTOL_Q: Field = 42950; // round(1e-5 * 2^32)
global ATOL_Q: Field = 43; // round(1e-8 * 2^32)

fn main(s1: [Field; 6], s2: [Field; 6]) -> pub Field {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(s1);
    noir_hw::check_inputs(s2);
    let mut count: Field = 0;
    for i in 0..6 {
        let tol = mul(abs(s2[i]), RTOL_Q) + ATOL_Q;
        if gt(abs(s1[i] - s2[i]), tol) {
            count += 1;
        }
    }
    count
}
