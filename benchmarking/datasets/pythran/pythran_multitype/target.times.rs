//! pythran_multitype__times (Pythran multitype), halo2-base baseline.
//! Kernel: `n * m` on two private integers, under the preconditions
//! `0 <= n <= 1024` and `0 <= m <= 1024` (`@requires`, asserted).
//! The precondition checks imply the int64 domain, so they replace the
//! window check of either input (CONVENTIONS 1.6); the product is exact.

use crate::prelude::*;

/// The bound of both preconditions (`@requires`), from the kernel's text.
const BOUND: u64 = 1024;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let n = inputs.int("n")?;
    let m = inputs.int("m")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // `n * m` assigns both inputs; the checks below are their only domain check.
    let n = ctx.load_witness(fe(n as i128));
    let m = ctx.load_witness(fe(m as i128));
    for x in [n, m] {
        q.range.check_less_than_safe(ctx, x, BOUND + 1);
    }
    let product = q.gate.mul(ctx, n, m);
    out.int(product);
    Ok(())
}
