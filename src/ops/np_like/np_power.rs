use super::define_np_arith;
define_np_arith!(NpPowerOp, "power", "np.power", ir_pow_i, ir_pow_f);
