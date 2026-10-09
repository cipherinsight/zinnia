// pythran_multitype__times: literal translation of kernel.py (LITERAL_BASELINES.md).
// Exact integer product. Halo2 standard input loaders replace forbidden local witnesses; source preconditions are asserted.

use crate::prelude::*;

/// The bound of both preconditions (`@requires`), from the kernel's text.
const BOUND: u64 = 1024;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let n = inputs.int("n")?; let m = inputs.int("m")?;
    let q = Q32::new(builder); let ctx = builder.main(0);
    let n = q.load_int(ctx, n); let m = q.load_int(ctx, m);
    for x in [n,m] { q.range.check_less_than_safe(ctx, x, BOUND+1); }
    let product = q.gate.mul(ctx, n, m);
    out.int(product); Ok(())
}
