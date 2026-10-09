// pythran_specialconvolve__specialconvolve: literal translation of kernel.py (LITERAL_BASELINES.md).
// Every elementwise intermediate is computed in full in expression order; integer arithmetic is exact.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const R: usize = 16; // = a_0
const C: usize = 16; // = a_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[R,C])?;
    let q = Q32::new(builder); let ctx = builder.main(0);
    let a = q.load_ints(ctx, &a);
    let mut row = Vec::with_capacity((R-2)*C);
    for i in 0..R-2 { for j in 0..C { row.push(q.add(ctx, a[(i+1)*C+j], a[i*C+j])); } }
    for i in 0..R-2 { for j in 0..C { row[i*C+j] = q.add(ctx, row[i*C+j], a[(i+2)*C+j]); } }
    let mut col = Vec::with_capacity((R-2)*(C-2));
    for i in 0..R-2 { for j in 0..C-2 { col.push(q.add(ctx, row[i*C+j+1], row[i*C+j])); } }
    for i in 0..R-2 { for j in 0..C-2 { col[i*(C-2)+j] = q.add(ctx, col[i*(C-2)+j], row[i*C+j+2]); } }
    let mut scaled = Vec::with_capacity((R-2)*(C-2));
    for i in 0..R-2 { for j in 0..C-2 { scaled.push(q.gate.mul(ctx, Constant(fe(9)), a[(i+1)*C+j+1])); } }
    for k in 0..col.len() { col[k] = q.sub(ctx, col[k], scaled[k]); }
    out.ints(&col); Ok(())
}
