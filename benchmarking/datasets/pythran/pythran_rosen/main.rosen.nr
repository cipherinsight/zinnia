// pythran_rosen__rosen (Pythran rosen), Noir baseline.
// Kernel: the Rosenbrock function of x: Float[N],
// np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).
// Each ** 2 is a rounded product (an elementwise array NumPy materializes);
// 100 * v is exact; the sum is exact.
// @outputs: float[1]

use noir_hw::{check_inputs, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0

fn main(x: [Field; N]) -> pub Field {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(x);
    let mut total: Field = 0;
    for i in 0..N - 1 {
        let gap = x[i + 1] - mul(x[i], x[i]);
        let rest = SCALE - x[i];
        total += 100 * mul(gap, gap) + mul(rest, rest);
    }
    total
}
