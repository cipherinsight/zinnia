use super::define_np_compare;
define_np_compare!(NpLessOp, "less", "np.less", ir_less_than_i, ir_less_than_f);
