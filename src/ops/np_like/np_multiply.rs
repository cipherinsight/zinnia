use super::define_np_arith;
define_np_arith!(NpMultiplyOp, "multiply", "np.multiply", ir_mul_i, ir_mul_f);
