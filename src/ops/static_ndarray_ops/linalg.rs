//! Static `np.linalg` verification helpers.
//!
//! These helpers intentionally do not compute inverses / solves / Cholesky
//! factors. The caller supplies the witness array, and the circuit verifies
//! the relation that makes that witness sound.

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

const IMPLICIT_LINALG_MSG: &str =
    "np.linalg.{op}: implicit result-returning linear algebra is deferred because \
     @zk_external currently returns one scalar, not an array witness; supply an \
     explicit witness and call np.linalg.verify_inverse/verify_solve/verify_cholesky";

fn shape_of(v: &Value) -> Vec<usize> {
    crate::helpers::composite::get_composite_shape(v)
}

fn dtype_of(v: &Value) -> Option<NumberType> {
    match v {
        Value::StaticArray { dtype, .. } => Some(*dtype),
        Value::List(_) | Value::Tuple(_) => {
            let flat = crate::helpers::composite::flatten_composite(v);
            if flat.iter().any(|x| matches!(x, Value::Complex { .. })) {
                Some(NumberType::Complex)
            } else if flat.iter().any(|x| matches!(x, Value::Float(_))) {
                Some(NumberType::Float)
            } else {
                Some(NumberType::Integer)
            }
        }
        _ => None,
    }
}

fn ensure_static_numeric(v: &Value, arg: &str, op: &str) {
    let dtype = dtype_of(v).unwrap_or_else(|| {
        panic!("np.linalg.{op}: {arg} must be a static numeric array; dynamic shapes are deferred")
    });
    if dtype == NumberType::Complex {
        panic!("np.linalg.{op}: Complex arrays are deferred for this first static numeric slice");
    }
}

fn assert_same_shape(lhs: &[usize], rhs: &[usize], what: &str, op: &str) {
    if lhs != rhs {
        panic!(
            "np.linalg.{op}: {what} shape mismatch: left {:?}, right {:?}",
            lhs, rhs
        );
    }
}

fn assert_all_equal(b: &mut IRBuilder, lhs: &Value, rhs: &Value, what: &str, op: &str) {
    let lhs_shape = shape_of(lhs);
    let rhs_shape = shape_of(rhs);
    assert_same_shape(&lhs_shape, &rhs_shape, what, op);
    let lhs = crate::helpers::static_array::base::deep_to_value_list(b, lhs);
    let rhs = crate::helpers::static_array::base::deep_to_value_list(b, rhs);
    let eq = crate::helpers::value_ops::apply_binary_op(b, "eq", &lhs, &rhs);
    let cond = crate::helpers::value_ops::to_scalar_bool(b, &eq);
    b.ir_assert(&cond);
}

fn cast_value_to_dtype(b: &mut IRBuilder, v: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => crate::helpers::value_ops::ensure_float(b, v),
        NumberType::Integer => match v {
            Value::Integer(_) | Value::Boolean(_) => v.clone(),
            _ => b.ir_int_cast(v),
        },
        NumberType::Complex => unreachable!("Complex linalg rejected upstream"),
    }
}

fn cast_to_dtype(b: &mut IRBuilder, v: &Value, dtype: NumberType) -> Value {
    match v {
        Value::StaticArray { shape, .. } => {
            let cells = crate::helpers::static_array::elementwise::payload_cells(b, v);
            let flat = cells
                .iter()
                .map(|cell| cast_value_to_dtype(b, cell, dtype))
                .collect();
            crate::helpers::static_array::base::build_static_array_from_flat(
                b,
                flat,
                shape.clone(),
                dtype,
            )
        }
        Value::List(data) => {
            let vals = data
                .values
                .iter()
                .map(|cell| cast_to_dtype(b, cell, dtype))
                .collect::<Vec<_>>();
            Value::List(crate::types::CompositeData {
                elements_type: vals.iter().map(|cell| cell.zinnia_type()).collect(),
                values: vals,
                value_id: crate::types::ValueId::next(),
            })
        }
        Value::Tuple(data) => {
            let vals = data
                .values
                .iter()
                .map(|cell| cast_to_dtype(b, cell, dtype))
                .collect::<Vec<_>>();
            Value::Tuple(crate::types::CompositeData {
                elements_type: vals.iter().map(|cell| cell.zinnia_type()).collect(),
                values: vals,
                value_id: crate::types::ValueId::next(),
            })
        }
        _ => cast_value_to_dtype(b, v, dtype),
    }
}

fn identity_matrix(b: &mut IRBuilder, n: usize, dtype: NumberType) -> Value {
    let zero = match dtype {
        NumberType::Float => b.ir_constant_float(0.0),
        NumberType::Integer => b.ir_constant_int(0),
        NumberType::Complex => unreachable!("Complex linalg rejected upstream"),
    };
    let one = match dtype {
        NumberType::Float => b.ir_constant_float(1.0),
        NumberType::Integer => b.ir_constant_int(1),
        NumberType::Complex => unreachable!("Complex linalg rejected upstream"),
    };
    let mut flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            flat.push(if i == j { one.clone() } else { zero.clone() });
        }
    }
    crate::helpers::static_array::base::build_static_array_from_flat(b, flat, vec![n, n], dtype)
}

fn require_square_matrix(v: &Value, arg: &str, op: &str) -> usize {
    let shape = shape_of(v);
    if shape.len() != 2 || shape[0] != shape[1] {
        panic!(
            "np.linalg.{op}: {arg} must be a square 2-D static array; got {:?}",
            shape
        );
    }
    shape[0]
}

pub fn np_linalg_implicit_result(op: &str) -> Value {
    panic!("{}", IMPLICIT_LINALG_MSG.replace("{op}", op));
}

pub fn np_linalg_verify_inverse(b: &mut IRBuilder, args: &[Value]) -> Value {
    if args.len() != 2 {
        panic!("np.linalg.verify_inverse: expected arguments (A, Y)");
    }
    let a = args
        .first()
        .unwrap_or_else(|| panic!("np.linalg.verify_inverse: A argument required"));
    let y = args
        .get(1)
        .unwrap_or_else(|| panic!("np.linalg.verify_inverse: witness inverse Y argument required"));
    ensure_static_numeric(a, "A", "verify_inverse");
    ensure_static_numeric(y, "Y", "verify_inverse");
    let n = require_square_matrix(a, "A", "verify_inverse");
    let y_shape = shape_of(y);
    if y_shape != vec![n, n] {
        panic!(
            "np.linalg.verify_inverse: Y must have shape [{n}, {n}] to match A; got {:?}",
            y_shape
        );
    }
    let dtype = if dtype_of(a) == Some(NumberType::Float) || dtype_of(y) == Some(NumberType::Float)
    {
        NumberType::Float
    } else {
        NumberType::Integer
    };
    let a_for_check = cast_to_dtype(b, a, dtype);
    let y_for_check = cast_to_dtype(b, y, dtype);
    let a_for_matmul = crate::helpers::static_array::base::deep_to_value_list(b, &a_for_check);
    let y_for_matmul = crate::helpers::static_array::base::deep_to_value_list(b, &y_for_check);
    let identity = identity_matrix(b, n, dtype);
    let ay = crate::ops::static_ndarray_ops::matmul(b, &a_for_matmul, &y_for_matmul);
    assert_all_equal(b, &ay, &identity, "A @ Y", "verify_inverse");
    let ya = crate::ops::static_ndarray_ops::matmul(b, &y_for_matmul, &a_for_matmul);
    assert_all_equal(b, &ya, &identity, "Y @ A", "verify_inverse");
    y.clone()
}

pub fn np_linalg_verify_solve(b: &mut IRBuilder, args: &[Value]) -> Value {
    if args.len() != 3 {
        panic!("np.linalg.verify_solve: expected arguments (A, x, b)");
    }
    let a = args
        .first()
        .unwrap_or_else(|| panic!("np.linalg.verify_solve: A argument required"));
    let x = args
        .get(1)
        .unwrap_or_else(|| panic!("np.linalg.verify_solve: solution witness x argument required"));
    let rhs = args
        .get(2)
        .unwrap_or_else(|| panic!("np.linalg.verify_solve: b argument required"));
    ensure_static_numeric(a, "A", "verify_solve");
    ensure_static_numeric(x, "x", "verify_solve");
    ensure_static_numeric(rhs, "b", "verify_solve");
    let n = require_square_matrix(a, "A", "verify_solve");
    let x_shape = shape_of(x);
    let b_shape = shape_of(rhs);
    match (x_shape.as_slice(), b_shape.as_slice()) {
        ([xn], [bn]) if *xn == n && *bn == n => {}
        ([xr, xc], [br, bc]) if *xr == n && *br == n && *xc == *bc => {}
        _ => panic!(
            "np.linalg.verify_solve: expected x and b to both be shape [{n}] or [{n}, k]; got x {:?}, b {:?}",
            x_shape, b_shape
        ),
    }
    let dtype = if dtype_of(a) == Some(NumberType::Float)
        || dtype_of(x) == Some(NumberType::Float)
        || dtype_of(rhs) == Some(NumberType::Float)
    {
        NumberType::Float
    } else {
        NumberType::Integer
    };
    let a_for_check = cast_to_dtype(b, a, dtype);
    let x_for_check = cast_to_dtype(b, x, dtype);
    let rhs_for_check = cast_to_dtype(b, rhs, dtype);
    let a_for_matmul = crate::helpers::static_array::base::deep_to_value_list(b, &a_for_check);
    let x_for_matmul = crate::helpers::static_array::base::deep_to_value_list(b, &x_for_check);
    let ax = crate::ops::static_ndarray_ops::matmul(b, &a_for_matmul, &x_for_matmul);
    assert_all_equal(b, &ax, &rhs_for_check, "A @ x vs b", "verify_solve");
    x.clone()
}

pub fn np_linalg_verify_cholesky(b: &mut IRBuilder, args: &[Value]) -> Value {
    if args.len() != 2 {
        panic!("np.linalg.verify_cholesky: expected arguments (A, L)");
    }
    let a = args
        .first()
        .unwrap_or_else(|| panic!("np.linalg.verify_cholesky: A argument required"));
    let l = args.get(1).unwrap_or_else(|| {
        panic!("np.linalg.verify_cholesky: lower-triangular witness L argument required")
    });
    ensure_static_numeric(a, "A", "verify_cholesky");
    ensure_static_numeric(l, "L", "verify_cholesky");
    let n = require_square_matrix(a, "A", "verify_cholesky");
    let l_shape = shape_of(l);
    if l_shape != vec![n, n] {
        panic!(
            "np.linalg.verify_cholesky: L must have shape [{n}, {n}] to match A; got {:?}",
            l_shape
        );
    }
    let dtype = if dtype_of(a) == Some(NumberType::Float) || dtype_of(l) == Some(NumberType::Float)
    {
        NumberType::Float
    } else {
        NumberType::Integer
    };
    let a_for_check = cast_to_dtype(b, a, dtype);
    let l_for_check = cast_to_dtype(b, l, dtype);
    let l_list = crate::helpers::static_array::base::to_value_list(b, &l_for_check);
    let zero_i = b.ir_constant_int(0);
    let zero_f = b.ir_constant_float(0.0);
    let l_dtype = dtype_of(&l_for_check).unwrap();
    let rows = match &l_list {
        Value::List(rows) | Value::Tuple(rows) => &rows.values,
        _ => unreachable!("2-D StaticArray materialises to nested list"),
    };
    for i in 0..n {
        let row = match &rows[i] {
            Value::List(row) | Value::Tuple(row) => &row.values,
            _ => panic!("np.linalg.verify_cholesky: L must be 2-D"),
        };
        for cell in row.iter().skip(i + 1) {
            let zero = if l_dtype == NumberType::Float {
                &zero_f
            } else {
                &zero_i
            };
            let eq = crate::helpers::value_ops::apply_binary_op(b, "eq", cell, zero);
            let cond = crate::helpers::value_ops::to_scalar_bool(b, &eq);
            b.ir_assert(&cond);
        }
    }
    let lt = crate::helpers::array_ops::transpose(b, &l_for_check, &[]);
    let l_for_matmul = crate::helpers::static_array::base::deep_to_value_list(b, &l_for_check);
    let lt_for_matmul = crate::helpers::static_array::base::deep_to_value_list(b, &lt);
    let llt = crate::ops::static_ndarray_ops::matmul(b, &l_for_matmul, &lt_for_matmul);
    assert_all_equal(b, &llt, &a_for_check, "L @ L.T vs A", "verify_cholesky");
    l.clone()
}
