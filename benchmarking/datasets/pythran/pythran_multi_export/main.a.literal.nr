// pythran_multi_export__a: literal translation of kernel.py (LITERAL_BASELINES.md).
// Identity return; only input-domain checking adds constraints.
// @outputs: float[1]
fn main(i: Field) -> pub Field { noir_hw::check_input(i); i }
