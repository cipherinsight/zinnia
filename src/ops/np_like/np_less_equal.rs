use super::define_np_compare;
define_np_compare!(NpLessEqualOp, "less_equal", "np.less_equal", ir_less_than_or_equal_i, ir_less_than_or_equal_f);
