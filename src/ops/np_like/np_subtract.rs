use super::define_np_arith;
define_np_arith!(NpSubtractOp, "subtract", "np.subtract", ir_sub_i, ir_sub_f);
