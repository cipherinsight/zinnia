//! DS-1000 case 0440: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Preserve enumerate order, full diagonal sqrt array, then slice addition to initialized zeros; off-diagonal inputs unread.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let y = inputs.floats("Y", &[10, 5, 5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let zero = q.float_const(ctx, 0.0);
    let mut x = vec![zero; 50];
    for i in 0..10 {
        // Only diagonal input elements are read by np.diag.
        let mat_diag: Vec<Cell> = (0..5).map(|j| q.load_float(ctx, y[i * 25 + j * 6])).collect();
        let diag: Vec<Cell> = mat_diag.iter().map(|v| q.sqrt(ctx, *v).cell).collect();
        for j in 0..5 { x[j * 10 + i] = q.add(ctx, x[j * 10 + i], diag[j]); }
    }
    out.floats(&x);
    Ok(())
}
