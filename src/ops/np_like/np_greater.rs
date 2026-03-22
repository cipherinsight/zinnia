use super::define_np_compare;
define_np_compare!(NpGreaterOp, "greater", "np.greater", ir_greater_than_i, ir_greater_than_f);
