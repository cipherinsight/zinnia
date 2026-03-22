use super::define_np_minmax;
define_np_minmax!(NpMinimumOp, "minimum", "np.minimum", ir_less_than_i, ir_less_than_f);
