use super::define_np_unary_math;
define_np_unary_math!(NpFAbsOp, "fabs", "np.fabs", ir_abs_f);
