//! pythran_insertion_sort__insertion_sort, halo2-base literal baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Source:
//!   for i in range(1, N):
//!       save = list2[i]
//!       j = i
//!       while j > 0 and list2[j - 1] > save:
//!           list2[j] = list2[j - 1]
//!           j -= 1
//!       list2[j] = save
//! Translation choices:
//! - The `while` loop's exit is data-dependent (L4): it runs at most `i`
//!   times (`j` falls from `i` to some value in `0..=i`), so it is unrolled
//!   to `i` fixed steps with `j = i - step`; within this range `j >= 1`
//!   always, so the `j > 0` half of the guard is statically true and only
//!   `list2[j - 1] > save` is evaluated each step, matching the source's
//!   short-circuit order.
//! - `active` is the flag that the while loop is still running; `shift` is
//!   `active and list2[j - 1] > save` (another step runs); `stop` is `active
//!   and not shift` (the exact step where the loop's condition first fails,
//!   i.e. where the loop's own `list2[j] = save` statement would run with
//!   this `j`). Every step writes `list2[j]`, gated by `shift`/`stop`, so a
//!   step that is neither keeps the slot's existing value (L4's "gates every
//!   iteration's effect").
//! - If `active` is still set after all `i` steps, the loop exited by
//!   `j == 0`, and the source's trailing `list2[j] = save` runs with `j = 0`.
//! - No sorting network and no hinted final position: this is the source's
//!   own shift-and-insert recurrence, direct.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 10; // = list2_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let values = inputs.floats("list2", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut values = q.load_floats_bounded(ctx, &values);

    for i in 1..N {
        // save = list2[i]; j = i
        let save = values[i];
        let mut active = ctx.load_constant(Fr::ONE);
        for step in 0..i {
            let j = i - step;
            // while j > 0 and list2[j - 1] > save: (j > 0 holds for every
            // step in this range)
            let larger = q.gt(ctx, values[j - 1], save);
            let shift = q.gate.and(ctx, active, larger);
            let stop = q.gate.sub(ctx, active, shift);
            // list2[j] = list2[j - 1] (only while still shifting)
            values[j] = q.select_i64(ctx, values[j - 1], values[j], shift);
            // list2[j] = save (the exact step the loop's guard first fails)
            values[j] = q.select_i64(ctx, save, values[j], stop);
            active = shift;
        }
        // Loop exited by j == 0: list2[0] = save.
        values[0] = q.select_i64(ctx, save, values[0], active);
    }
    let result: Vec<Cell> = values.iter().map(|v| v.cell).collect();
    out.floats(&result);
    Ok(())
}
