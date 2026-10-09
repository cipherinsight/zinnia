// pythran_ln__ln: literal translation of kernel.py (LITERAL_BASELINES.md).
// Local hint-free power helper preserves optimized MSB-first Q32 schedule; no powers or X-1 arrays shared.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = X_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("X", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let mut result: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    // (X - 1) ** 2 / 2
    let delta: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    let power: Vec<Cell> = delta.iter().map(|v| power_msb(&q,ctx,*v,2)).collect();
    let term: Vec<Cell> = power.iter().map(|v| q.floor_div_const(ctx,*v,2).0.cell).collect();
    result = (0..N).map(|i| q.sub(ctx,result[i],term[i])).collect();
    // (X - 1) ** 3 / 3
    let delta: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    let power: Vec<Cell> = delta.iter().map(|v| power_msb(&q,ctx,*v,3)).collect();
    let term: Vec<Cell> = power.iter().map(|v| q.floor_div_const(ctx,*v,3).0.cell).collect();
    result = (0..N).map(|i| q.add(ctx,result[i],term[i])).collect();
    // (X - 1) ** 4 / 4
    let delta: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    let power: Vec<Cell> = delta.iter().map(|v| power_msb(&q,ctx,*v,4)).collect();
    let term: Vec<Cell> = power.iter().map(|v| q.floor_div_const(ctx,*v,4).0.cell).collect();
    result = (0..N).map(|i| q.sub(ctx,result[i],term[i])).collect();
    // (X - 1) ** 5 / 5
    let delta: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    let power: Vec<Cell> = delta.iter().map(|v| power_msb(&q,ctx,*v,5)).collect();
    let term: Vec<Cell> = power.iter().map(|v| q.floor_div_const(ctx,*v,5).0.cell).collect();
    result = (0..N).map(|i| q.add(ctx,result[i],term[i])).collect();
    // (X - 1) ** 6 / 6
    let delta: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    let power: Vec<Cell> = delta.iter().map(|v| power_msb(&q,ctx,*v,6)).collect();
    let term: Vec<Cell> = power.iter().map(|v| q.floor_div_const(ctx,*v,6).0.cell).collect();
    result = (0..N).map(|i| q.sub(ctx,result[i],term[i])).collect();
    // (X - 1) ** 7 / 7
    let delta: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    let power: Vec<Cell> = delta.iter().map(|v| power_msb(&q,ctx,*v,7)).collect();
    let term: Vec<Cell> = power.iter().map(|v| q.floor_div_const(ctx,*v,7).0.cell).collect();
    result = (0..N).map(|i| q.add(ctx,result[i],term[i])).collect();
    // (X - 1) ** 8 / 8
    let delta: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    let power: Vec<Cell> = delta.iter().map(|v| power_msb(&q,ctx,*v,8)).collect();
    let term: Vec<Cell> = power.iter().map(|v| q.floor_div_const(ctx,*v,8).0.cell).collect();
    result = (0..N).map(|i| q.sub(ctx,result[i],term[i])).collect();
    // (X - 1) ** 9 / 9
    let delta: Vec<Cell> = x.iter().map(|v| q.sub(ctx,*v,Constant(fe(SCALE)))).collect();
    let power: Vec<Cell> = delta.iter().map(|v| power_msb(&q,ctx,*v,9)).collect();
    let term: Vec<Cell> = power.iter().map(|v| q.floor_div_const(ctx,*v,9).0.cell).collect();
    result = (0..N).map(|i| q.add(ctx,result[i],term[i])).collect();
    out.floats(&result); Ok(())
}
fn power_msb(q: &Q32,ctx: &mut Ctx,x: Cell,n: u32) -> Cell {
    let mut value = x;
    let top = 31-n.leading_zeros();
    for bit in (0..top).rev() {
        value = q.mul(ctx,value,value).cell;
        if (n>>bit)&1 != 0 { value=q.mul(ctx,value,x).cell; }
    }
    value
}
