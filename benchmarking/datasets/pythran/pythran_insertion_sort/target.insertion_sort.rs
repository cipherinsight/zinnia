//! pythran_insertion_sort__insertion_sort (Pythran insertion_sort), halo2-base baseline.
//! Kernel: insertion sort of `list2: Float[N]` in place; the output is
//! `list2` after the call (`mutated_outputs`), i.e. `np.sort(list2)`.
//! Only the sorted values are output, so the order among equal values does
//! not matter: a sorting network on the loaded window offsets (`q.sort`,
//! Batcher's merge exchange, 31 compare-exchanges for 10 values), each
//! input's window proof reused by the comparisons.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 10; // = list2_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let list2 = inputs.floats("list2", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let list2 = q.load_floats_bounded(ctx, &list2);
    let sorted: Vec<Cell> = q.sort(ctx, &list2).iter().map(|x| x.cell).collect();
    out.floats(&sorted);
    Ok(())
}
