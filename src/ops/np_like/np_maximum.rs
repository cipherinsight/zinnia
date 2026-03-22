use super::define_np_minmax;
define_np_minmax!(NpMaximumOp, "maximum", "np.maximum", ir_greater_than_i, ir_greater_than_f);
