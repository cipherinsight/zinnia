use super::define_np_arith;
define_np_arith!(NpPowOp, "pow", "np.pow", ir_pow_i, ir_pow_f);
