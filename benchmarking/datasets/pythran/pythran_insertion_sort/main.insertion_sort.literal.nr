// pythran_insertion_sort__insertion_sort, Noir literal baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Source:
//   for i in range(1, N):
//       save = list2[i]
//       j = i
//       while j > 0 and list2[j - 1] > save:
//           list2[j] = list2[j - 1]
//           j -= 1
//       list2[j] = save
// Translation choices:
// - The while loop's exit is data-dependent (L4): it runs at most i times,
//   unrolled to i fixed steps with j = i - step; within this range j >= 1
//   always, so only list2[j - 1] > save is evaluated each step (j > 0 holds
//   statically), matching the source's short-circuit order.
// - active is "the while loop is still running"; each step either shifts
//   (list2[j] = list2[j - 1], j -= 1) while active and list2[j - 1] > save,
//   or, on the exact step the guard first fails, writes save at this j and
//   clears active; a step that does neither leaves the slot unchanged.
// - If active is still set after all i steps, the loop exited by j == 0, and
//   list2[0] = save runs.
// - No sorting network and no hinted final position: the source's own
//   shift-and-insert recurrence, direct.
// @outputs: float[N]

use noir_hw::window;

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = list2_0

fn main(list2: [Field; N]) -> pub [Field; N] {
    let mut values = list2;
    let mut windows: [u64; N] = [0; N];
    for i in 0..N {
        windows[i] = window(values[i]);
    }
    for i in 1..N {
        // save = list2[i]; j = i
        let save = values[i];
        let save_window = windows[i];
        let mut active = true;
        for step in 0..i {
            let j = i - step;
            // while j > 0 and list2[j - 1] > save: (j > 0 holds for every
            // step in this range)
            let shift = active & (windows[j - 1] > save_window);
            if shift {
                // list2[j] = list2[j - 1]
                values[j] = values[j - 1];
                windows[j] = windows[j - 1];
            } else if active {
                // list2[j] = save (the exact step the loop's guard first fails)
                values[j] = save;
                windows[j] = save_window;
            }
            active = shift;
        }
        // Loop exited by j == 0: list2[0] = save.
        if active {
            values[0] = save;
            windows[0] = save_window;
        }
    }
    values
}
