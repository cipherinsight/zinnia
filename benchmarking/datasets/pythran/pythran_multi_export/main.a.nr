// pythran_multi_export__a (Pythran multi_export), Noir baseline.
// Kernel: `return i` on a scalar i: Float. Data movement only: the output is
// the input, with its domain check (NPB_BASELINES.md, section 5, rule 6).
// @outputs: float[1]

use noir_hw::checked;

fn main(i: Field) -> pub Field {
    checked(i)
}
