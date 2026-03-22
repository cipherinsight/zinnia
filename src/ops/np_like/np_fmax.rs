use super::define_np_minmax;
define_np_minmax!(NpFMaxOp, "fmax", "np.fmax", ir_greater_than_i, ir_greater_than_f);
