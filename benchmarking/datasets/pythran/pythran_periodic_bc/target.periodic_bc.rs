//! pythran_periodic_bc__periodic_bc (Pythran periodic_bc), halo2-base baseline.
//! Kernel: `f[:, 0, :] = f[:, -2, :]` on `f: Float[N0, N1, N2]`, returning
//! `f`. Data movement only: the plane `f[:, 0, :]` is overwritten before it
//! is read, so it is not loaded (unless it is the source plane itself); every
//! other element is loaded with its domain check and output as loaded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N0: usize = 3; // = f_0
const N1: usize = 3; // = f_1
const N2: usize = 3; // = f_2
/// The plane `-2` names (NumPy's wrap of a negative index).
const SOURCE: usize = N1 - 2;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let f = inputs.floats("f", &[N0, N1, N2])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let at = |i: usize, j: usize, k: usize| (i * N1 + j) * N2 + k;
    let read = |j: usize| j != 0 || SOURCE == 0;
    let mut loaded: Vec<Option<Cell>> = vec![None; N0 * N1 * N2];
    for i in 0..N0 {
        for j in (0..N1).filter(|j| read(*j)) {
            for k in 0..N2 {
                loaded[at(i, j, k)] = Some(q.load_float(ctx, f[at(i, j, k)]));
            }
        }
    }
    let mut result: Vec<Cell> = Vec::with_capacity(N0 * N1 * N2);
    for i in 0..N0 {
        for j in 0..N1 {
            let from = if j == 0 { SOURCE } else { j };
            for k in 0..N2 {
                result.push(loaded[at(i, from, k)].expect("read plane"));
            }
        }
    }
    out.floats(&result);
    Ok(())
}
