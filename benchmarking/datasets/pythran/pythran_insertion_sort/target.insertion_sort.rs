//! Source insertion_sort algorithm: no sorting network or hinted permutation.
use crate::prelude::*;
// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 10; // = list2_0
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let values = inputs.floats("list2", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut values = q.load_floats_bounded(ctx, &values);
    for i in 1..N {
        let save = values[i];
        let mut active = ctx.load_constant(Fr::ONE);
        // While active, j = i - step by induction: the body decrements j
        // once per step. After the first false guard no more writes occur.
        for step in 0..i {
            let j = i - step;
            let larger = q.gt(ctx, values[j - 1], save);
            let shift = q.gate.and(ctx, active, larger);
            let stop = q.gate.sub(ctx, active, shift);
            values[j] = q.select_i64(ctx, values[j - 1], values[j], shift);
            values[j] = q.select_i64(ctx, save, values[j], stop);
            active = shift;
        }
        values[0] = q.select_i64(ctx, save, values[0], active);
    }
    let result: Vec<Cell> = values.iter().map(|v| v.cell).collect();
    out.floats(&result);
    Ok(())
}
