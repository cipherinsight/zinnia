//! Polynomials: `np.polyval` and the Vandermonde matrix `np.vander`.

use crate::builder::IRBuilder;
use crate::types::Value;

use super::operand::{build_array, cell_op, one, StaticOperand};

/// `np.polyval(p, x)`: the polynomial with coefficients `p` (highest power
/// first) at `x`, by Horner's rule as NumPy evaluates it:
/// `y = zeros_like(x)`, then `y = y * x + c` for every coefficient `c`.
/// `x` is a scalar or an array of any shape; the result has `x`'s shape.
pub fn np_polyval(b: &mut IRBuilder, p: &Value, x: &Value) -> Value {
    let p = StaticOperand::read_numeric(b, p, "np.polyval");
    if p.shape.len() != 1 {
        panic!("np.polyval: the coefficients must be a 1-D array, got shape {:?}", p.shape);
    }
    // NumPy reads `x` as an array (`asanyarray`): a list must not pick up
    // Python's sequence arithmetic.
    let x = match x {
        Value::List(_) | Value::Tuple(_) => crate::helpers::static_array::base::to_static_array(b, x)
            .unwrap_or_else(|| panic!("np.polyval: x must be a number or a rectangular numeric array")),
        other => other.clone(),
    };
    let x = &x;
    if b.is_bool_cell(x) || crate::helpers::array_ops::is_bool_array(b, x) {
        panic!("np.polyval: boolean arrays are not supported (NumPy's arithmetic on them is logical)");
    }
    let apply = crate::helpers::value_ops::apply_binary_op;
    // `zeros_like(x) * x + p[0]` is `x * 0 + p[0]`: zero times any value is
    // exactly zero in both integer and fixed-point arithmetic.
    let zero = b.ir_constant_int(0);
    let mut y = apply(b, "mul", x, &zero);
    for (k, c) in p.cells.iter().enumerate() {
        if k > 0 {
            y = apply(b, "mul", &y, x);
        }
        y = apply(b, "add", &y, c);
    }
    y
}

/// `np.vander(x, N=None, increasing=False)`: the matrix whose columns are
/// powers of the 1-D array `x`, from `x**(N-1)` down to `x**0` (up from
/// `x**0` when `increasing`). `N` defaults to `len(x)`. Each power is the
/// previous one times `x`, NumPy's running product.
pub fn np_vander(b: &mut IRBuilder, x: &Value, n: Option<usize>, increasing: bool) -> Value {
    let x = StaticOperand::read_numeric(b, x, "np.vander");
    if x.shape.len() != 1 {
        panic!("np.vander: x must be a one-dimensional array, got shape {:?}", x.shape);
    }
    let columns = n.unwrap_or(x.cells.len());
    let mut cells = Vec::with_capacity(x.cells.len() * columns);
    for value in &x.cells {
        // powers[k] = value ** k
        let mut powers: Vec<Value> = Vec::with_capacity(columns);
        for k in 0..columns {
            let power = match k {
                0 => one(b, x.dtype),
                1 => value.clone(),
                _ => cell_op(b, "mul", &powers[k - 1], value),
            };
            powers.push(power);
        }
        if !increasing {
            powers.reverse();
        }
        cells.extend(powers);
    }
    build_array(b, cells, vec![x.cells.len(), columns], x.dtype)
}
