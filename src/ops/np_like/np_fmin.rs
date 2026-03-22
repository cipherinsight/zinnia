use super::define_np_minmax;
define_np_minmax!(NpFMinOp, "fmin", "np.fmin", ir_less_than_i, ir_less_than_f);
