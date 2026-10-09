// pythran_bubble_sort__bubble_sort: literal translation of kernel.py (LITERAL_BASELINES.md).
// Source bubble loops, fresh bounds per comparison. swap_test=True is unconditional, so break depends only on loop shape.

use crate::prelude::*;
// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 11; // = list0_0
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let list0 = inputs.ints("list0", &[N])?;
    let q = Q32::new(builder); let ctx = builder.main(0);
    let list0 = q.load_ints(ctx, &list0);
    let mut list1 = list0.clone();
    for i in 0..N-1 {
        let mut swap_test = false;
        for j in 0..N-i-1 {
            let left = q.bound(ctx, list1[j]);
            let right = q.bound(ctx, list1[j+1]);
            let swap = q.gt(ctx, left, right);
            let old_left = list1[j]; let old_right = list1[j+1];
            list1[j] = q.gate.select(ctx, old_right, old_left, swap);
            list1[j+1] = q.gate.select(ctx, old_left, old_right, swap);
            swap_test = true;
        }
        if !swap_test { break; }
    }
    out.ints(&list1); Ok(())
}
