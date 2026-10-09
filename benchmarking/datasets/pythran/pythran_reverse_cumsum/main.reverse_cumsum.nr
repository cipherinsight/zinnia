// pythran_reverse_cumsum__reverse_cumsum (Pythran reverse_cumsum), Noir baseline.
// Kernel: `np.cumsum(x[::-1])[::-1]` on x: Float[N]: element i is
// x[i] + ... + x[N - 1], the suffix sums, built from the end.
// @outputs: float[N]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = x_0

fn main(x: [Field; N]) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(x);
    let mut sums = x;
    for k in 1..N {
        let i = N - 1 - k;
        sums[i] = x[i] + sums[i + 1];
    }
    sums
}
