use super::define_np_arith;
define_np_arith!(NpModOp, "mod", "np.mod", ir_mod_i, ir_mod_f);
