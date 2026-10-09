// Source bubble_sort algorithm; no hinted permutation or sorting network.
// @outputs: int[N]
use noir_hw::window;
// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 11; // = list0_0
fn main(list0: [Field; N]) -> pub [Field; N] {
    let mut values = list0;
    let mut windows: [u64; N] = [0; N];
    for i in 0..N { windows[i] = window(values[i]); }
    // swap_test=True is unconditional in the source inner body; every
    // outer pass has a comparison, so its break is unreachable.
    for i in 0..N - 1 {
        for j in 0..N - i - 1 {
            if windows[j] > windows[j + 1] {
                let left = values[j];
                let left_window = windows[j];
                values[j] = values[j + 1]; windows[j] = windows[j + 1];
                values[j + 1] = left; windows[j + 1] = left_window;
            }
        }
    }
    values
}
