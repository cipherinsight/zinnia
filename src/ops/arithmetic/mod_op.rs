use super::define_binary_arith_op;
define_binary_arith_op!(ModOp, "mod", ir_mod_i, ir_mod_f);
