//! DS-1000 case 0328: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Shape condition is constant. Full copy and slice assignment are wire aliases; only retained inputs need domain checks (L7).
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let length = 3;
    // Constant false branch of length > A.shape[0]. Copy all entries,
    // then overwrite the tail, before loading the retained private wires.
    let mut result = a.to_vec();
    for i in length..5 { result[i] = 0.0; }
    let retained = q.load_floats(ctx, &result[..length]);
    out.floats(&retained);
    out.const_floats(&[0.0; 2]);
    Ok(())
}
