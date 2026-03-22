use super::define_np_unary_math;
define_np_unary_math!(NpAbsoluteOp, "absolute", "np.absolute", ir_abs_f);
