use super::define_np_unary_math;
define_np_unary_math!(NpSqrtOp, "sqrt", "np.sqrt", ir_sqrt_f);
