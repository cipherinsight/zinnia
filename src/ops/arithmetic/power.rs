use super::define_binary_arith_op;
define_binary_arith_op!(PowerOp, "power", ir_pow_i, ir_pow_f);
