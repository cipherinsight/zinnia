use super::define_np_arith;
define_np_arith!(NpAddOp, "add", "np.add", ir_add_i, ir_add_f);
