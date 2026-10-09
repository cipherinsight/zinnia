// pythran_weights__weights: literal translation of kernel.py (LITERAL_BASELINES.md).
// Equality array and each reduction are separate; repeat threshold*length each iteration; float32 stays Q32.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const S: usize = 5; // = input_data_0
const L: usize = 2; // = input_data_1
const COUNT_BITS: usize = 3; // = bits(input_data_0)

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.ints("input_data", &[S, L])?;
    let threshold = inputs.float("threshold")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_ints(ctx, &data);
    let threshold = q.load_float(ctx, threshold);
    let mut weights = vec![ctx.load_zero(); S];
    for i in 0..S {
        let vector = &data[i*L..(i+1)*L];
        let mut matches = vec![ctx.load_zero(); S*L];
        for row in 0..S { for c in 0..L { matches[row*L+c] = q.eq(ctx,vector[c],data[row*L+c]); } }
        let mut counts = vec![ctx.load_zero(); S];
        for row in 0..S { for c in 0..L { counts[row] = q.add(ctx,counts[row],matches[row*L+c]); } }
        let bar = q.mul_int(ctx,threshold,Constant(fe(L as i128)));
        let mut over = Vec::new();
        for row in 0..S {
            let left = q.int_to_float(ctx,counts[row]);
            let left = q.bound(ctx,left); let right = q.bound(ctx,bar);
            over.push(q.gt(ctx,left,right));
        }
        let mut total = ctx.load_zero();
        for v in over { total = q.add(ctx,total,v); }
        let denominator = q.int_to_float(ctx,total);
        let denominator = q.bound(ctx,denominator);
        weights[i] = q.div(ctx,Constant(fe(SCALE)),denominator).cell;
    }
    out.floats(&weights); Ok(())
}
