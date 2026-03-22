use super::define_np_arith;
define_np_arith!(NpFloorDivideOp, "floor_divide", "np.floor_divide", ir_floor_div_i, ir_floor_div_f);
