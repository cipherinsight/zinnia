use super::define_np_compare;
define_np_compare!(NpEqualOp, "equal", "np.equal", ir_equal_i, ir_equal_f);
