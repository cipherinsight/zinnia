//! DS-1000 case 0438: literal translation of kernel.py (LITERAL_BASELINES.md).
//! corrcoef uses the full sample covariance matrix, dot rescale once, /3, row/column sqrt normalization and clipping.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let pv = inputs.ints("post", &[4])?;
    let dv = inputs.ints("distance", &[4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let post = q.load_ints(ctx, &pv);
    let distance = q.load_ints(ctx, &dv);
    // corrcoef: stack and cast, row means, centered covariance matrix.
    let mut data = Vec::new();
    for source in [&post, &distance] {
        let mut row = Vec::new();
        for x in source { row.push(q.int_to_float(ctx, *x)); }
        data.push(row);
    }
    let mut means = Vec::new();
    for row in &data {
        let mut sum = q.int_const(ctx, 0);
        for x in row { sum = q.add(ctx, sum, *x); }
        means.push(q.floor_div_const(ctx, sum, 4).0.cell);
    }
    let centered: Vec<Vec<Cell>> = (0..2).map(|i| data[i].iter().map(|x| q.sub(ctx, *x, means[i])).collect()).collect();
    let mut covariance = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            let mut sum = q.int_const(ctx, 0);
            for k in 0..4 {
                let product = q.gate.mul(ctx, centered[i][k], centered[j][k]);
                sum = q.add(ctx, sum, product);
            }
            let dot = q.rescale(ctx, sum).cell;
            covariance.push(q.floor_div_const(ctx, dot, 3).0.cell);
        }
    }
    let diagonal = [covariance[0], covariance[3]];
    let stddev: Vec<Cell> = diagonal.iter().map(|x| q.sqrt(ctx, *x).cell).collect();
    // Normalize rows, then columns, keeping all four entries.
    for i in 0..2 {
        for j in 0..2 {
            let divisor = q.bound(ctx, stddev[i]);
            covariance[i * 2 + j] = q.div(ctx, covariance[i * 2 + j], divisor).cell;
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let divisor = q.bound(ctx, stddev[j]);
            covariance[i * 2 + j] = q.div(ctx, covariance[i * 2 + j], divisor).cell;
        }
    }
    // Real corrcoef clips every matrix element to [-1, 1].
    for value in &mut covariance {
        let v = q.bound(ctx, *value);
        let low = q.float_const(ctx, -1.0);
        let low = q.bound(ctx, low);
        let clipped = q.max(ctx, v, low).cell;
        let clipped = q.bound(ctx, clipped);
        let high = q.float_const(ctx, 1.0);
        let high = q.bound(ctx, high);
        *value = q.min(ctx, clipped, high).cell;
    }
    out.float(covariance[1]);
    Ok(())
}
