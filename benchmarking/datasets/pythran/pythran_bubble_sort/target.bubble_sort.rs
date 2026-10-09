//! pythran_bubble_sort__bubble_sort (Pythran bubble_sort), halo2-base baseline.
//! Kernel: bubble sort of a copy of `list0: Int[N]`, returned as a list,
//! i.e. `sorted(list0)` (the early exit never fires: `swap_test` is set on
//! every pass).
//! Only the sorted values are output, so the order among equal values does
//! not matter: a sorting network on the loaded window offsets (`q.sort`,
//! Batcher's merge exchange), each input's window proof reused by the
//! comparisons.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 11; // = list0_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let list0 = inputs.ints("list0", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let list0 = q.load_ints_bounded(ctx, &list0);
    let sorted: Vec<Cell> = q.sort(ctx, &list0).iter().map(|x| x.cell).collect();
    out.ints(&sorted);
    Ok(())
}
