use super::define_np_compare;
define_np_compare!(NpGreaterEqualOp, "greater_equal", "np.greater_equal", ir_greater_than_or_equal_i, ir_greater_than_or_equal_f);
