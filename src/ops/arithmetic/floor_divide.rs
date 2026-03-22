use super::define_binary_arith_op;
define_binary_arith_op!(FloorDivideOp, "floor_divide", ir_floor_div_i, ir_floor_div_f);
