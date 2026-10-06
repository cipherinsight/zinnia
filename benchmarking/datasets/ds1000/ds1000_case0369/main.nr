// DS-1000 case 0369, Noir baseline.
// Kernel: `np.isclose(a, a[:, 0].reshape(-1, 1), atol=0).all()` on Float[3, 3]:
// every column close to column 0 by NumPy's |x - y| <= atol + rtol * |y| with
// rtol = 1e-5, atol = 0, y = a[i, 0]. Column 0 against itself is always close
// and is skipped; the tolerance rtol * |a[i, 0]| (a rescaled product) is
// computed once per row.
// @outputs: bool[1]

use noir_hw::{abs, mul, window};

global N: u32 = 3;
global RTOL_Q: Field = 42950; // round(1e-5 * 2^32)

fn main(a: [Field; 9]) -> pub bool {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut all_close = true;
    for i in 0..N {
        let y = a[i * N];
        let tol = window(mul(abs(y), RTOL_Q));
        for j in 1..N {
            let gap = window(abs(a[i * N + j] - y));
            all_close &= gap <= tol;
        }
    }
    all_close
}
