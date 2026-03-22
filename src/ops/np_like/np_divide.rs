use super::define_np_arith;
define_np_arith!(NpDivideOp, "divide", "np.divide", ir_div_i, ir_div_f);
