use super::define_np_unary_math;
define_np_unary_math!(NpTanHOp, "tanh", "np.tanh", ir_tanh_f);
