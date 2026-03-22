use super::define_np_arith;
define_np_arith!(NpFModOp, "fmod", "np.fmod", ir_mod_i, ir_mod_f);
