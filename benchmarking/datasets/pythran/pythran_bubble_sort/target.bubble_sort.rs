//! Source bubble_sort algorithm: no sorting network or hinted permutation.
use crate::prelude::*;
// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 11; // = list0_0
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let values = inputs.ints("list0", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut values = q.load_ints_bounded(ctx, &values);
    // Source swap_test=True is unconditional in each inner iteration;
    // every outer pass has at least one comparison, so break never fires.
    for i in 0..N.saturating_sub(1) {
        for j in 0..N - i - 1 {
            let left = values[j];
            let right = values[j + 1];
            let swap = q.gt(ctx, left, right);
            values[j] = q.select_i64(ctx, right, left, swap);
            values[j + 1] = q.select_i64(ctx, left, right, swap);
        }
    }
    let result: Vec<Cell> = values.iter().map(|v| v.cell).collect();
    out.ints(&result);
    Ok(())
}
