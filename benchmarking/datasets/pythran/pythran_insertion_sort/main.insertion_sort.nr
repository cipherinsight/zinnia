// Source insertion_sort algorithm; no hinted permutation or sorting network.
// @outputs: float[N]
use noir_hw::window;
// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = list2_0
fn main(list2: [Field; N]) -> pub [Field; N] {
    let mut values = list2;
    let mut windows: [u64; N] = [0; N];
    for i in 0..N { windows[i] = window(values[i]); }
    for i in 1..N {
        let save = values[i];
        let save_window = windows[i];
        let mut active = true;
        // While active j=i-step; after the first false guard no writes.
        for step in 0..i {
            let j = i - step;
            let shift = active & (windows[j - 1] > save_window);
            if shift {
                values[j] = values[j - 1];
                windows[j] = windows[j - 1];
            } else if active {
                values[j] = save;
                windows[j] = save_window;
            }
            active = shift;
        }
        if active { values[0] = save; windows[0] = save_window; }
    }
    values
}
