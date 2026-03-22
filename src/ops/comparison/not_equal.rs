use super::define_compare_op;
define_compare_op!(NotEqualOp, "ne", ir_not_equal_i, ir_not_equal_f);
