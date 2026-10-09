// pythran_reverse_cumsum__reverse_cumsum: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reverse is wire movement; cumsum is a running accumulator in reversed index order.
// @outputs: float[N]
global N: u32 = 10; // = x_0

fn main(x: [Field; N]) -> pub [Field; N] {
    noir_hw::check_inputs(x);
    let mut reverse = [0; N];
    for i in 0..N {
        reverse[i] = x[N - 1 - i];
    }
    let mut cumulative = [0; N];
    let mut sum: Field = 0;
    for i in 0..N {
        sum += reverse[i];
        cumulative[i] = sum;
    }
    let mut result = [0; N];
    for i in 0..N {
        result[i] = cumulative[N - 1 - i];
    }
    result
}
