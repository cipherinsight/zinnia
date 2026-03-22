use super::define_np_compare;
define_np_compare!(NpNotEqualOp, "not_equal", "np.not_equal", ir_not_equal_i, ir_not_equal_f);
