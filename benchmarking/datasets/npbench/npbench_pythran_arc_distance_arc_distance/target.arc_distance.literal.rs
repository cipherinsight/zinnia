//! npbench_pythran_arc_distance_arc_distance__arc_distance: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Each array operation completes in expression order; shared canonical transcendental arithmetic.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = theta_1_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let theta_1 = inputs.floats("theta_1", &[N])?;
    let phi_1 = inputs.floats("phi_1", &[N])?;
    let theta_2 = inputs.floats("theta_2", &[N])?;
    let phi_2 = inputs.floats("phi_2", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let theta_1 = q.load_floats(ctx, &theta_1);
    let phi_1 = q.load_floats(ctx, &phi_1);
    let theta_2 = q.load_floats(ctx, &theta_2);
    let phi_2 = q.load_floats(ctx, &phi_2);
    let dt: Vec<Cell> = (0..N).map(|i| q.sub(ctx, theta_2[i],theta_1[i])).collect();
    let ht: Vec<Cell> = dt.iter().map(|v| q.floor_div_const(ctx,*v,2).0.cell).collect();
    let st: Vec<Cell> = ht.iter().map(|v| { let v=q.bound(ctx,*v); q.canonical_sin(ctx,v) }).collect();
    let lat: Vec<Cell> = st.iter().map(|v| q.pow_const(ctx,*v,2)).collect();
    let c1: Vec<Cell> = theta_1.iter().map(|v| q.canonical_cos(ctx,*v)).collect();
    let c2: Vec<Cell> = theta_2.iter().map(|v| q.canonical_cos(ctx,*v)).collect();
    let cc: Vec<Cell> = (0..N).map(|i| q.mul(ctx,c1[i],c2[i]).cell).collect();
    let dp: Vec<Cell> = (0..N).map(|i| q.sub(ctx,phi_2[i],phi_1[i])).collect();
    let hp: Vec<Cell> = dp.iter().map(|v| q.floor_div_const(ctx,*v,2).0.cell).collect();
    let sp: Vec<Cell> = hp.iter().map(|v| { let v=q.bound(ctx,*v); q.canonical_sin(ctx,v) }).collect();
    let lon: Vec<Cell> = sp.iter().map(|v| q.pow_const(ctx,*v,2)).collect();
    let weighted: Vec<Cell> = (0..N).map(|i| q.mul(ctx,cc[i],lon[i]).cell).collect();
    let temp: Vec<Cell> = (0..N).map(|i| q.add(ctx,lat[i],weighted[i])).collect();
    let y: Vec<Cell> = temp.iter().map(|v| q.sqrt(ctx,*v).cell).collect();
    let rest: Vec<Cell> = temp.iter().map(|v| q.sub(ctx,Constant(fe(SCALE)),*v)).collect();
    let x: Vec<Cell> = rest.iter().map(|v| q.sqrt(ctx,*v).cell).collect();
    let angles: Vec<Cell> = (0..N).map(|i| {
        let y=q.bound(ctx,y[i]); let x=q.bound(ctx,x[i]); q.canonical_arctan2(ctx,y,x)
    }).collect();
    let result: Vec<Cell> = angles.iter().map(|v| q.mul_int(ctx,*v,Constant(fe(2)))).collect();
    out.floats(&result); Ok(())
}
