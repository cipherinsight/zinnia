//! `np.heaviside(x1, x2)`: the Heaviside step function of `x1`, with the
//! value `x2` at `x1 == 0`. The result is a float (NumPy has float loops
//! only); arrays broadcast like any binary ufunc.

use crate::builder::IRBuilder;
use crate::types::Value;

impl IRBuilder {
    /// `0` where `x1 < 0`, `x2` where `x1 == 0`, `1` where `x1 > 0`, on two
    /// Float scalars.
    pub(crate) fn heaviside_f(&mut self, x1: &Value, x2: &Value) -> Value {
        let zero = self.ir_constant_float(0.0);
        let one = self.ir_constant_float(1.0);
        let negative = self.ir_less_than_f(x1, &zero);
        let at_zero = self.ir_equal_f(x1, &zero);
        let non_negative = self.ir_select_f(&at_zero, x2, &one);
        self.ir_select_f(&negative, &zero, &non_negative)
    }
}

define_np_binary_math!(NpHeavisideOp, "heaviside", "np.heaviside", heaviside_f);
