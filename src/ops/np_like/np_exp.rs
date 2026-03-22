use super::define_np_unary_math;
define_np_unary_math!(NpExpOp, "exp", "np.exp", ir_exp_f);
