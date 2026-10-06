//! Argument binding for `np.linalg.norm(x, ord=None, axis=None, keepdims=False)`.
//!
//! `ord` is read from the AST rather than visited: NumPy's `inf` orders are
//! written `np.inf` / `-np.inf` (or `math.inf`, `float('inf')`), and an
//! infinity is not a value a circuit can hold. Every other argument is
//! visited as usual and must be a compile-time constant where NumPy needs
//! one to decide the result's shape.

use crate::ast::*;
use crate::ops::static_ndarray_ops::{NormCall, NormOrder};
use crate::types::Value;

use super::super::IRGenerator;

const PARAMETERS: [&str; 4] = ["x", "ord", "axis", "keepdims"];

impl IRGenerator {
    pub(super) fn visit_np_linalg_norm(&mut self, n: &ASTExprAttribute) -> Value {
        if n.args.iter().any(|a| matches!(a, ASTNode::ASTStarredExpr(_))) {
            panic!("np.linalg.norm: starred arguments are not supported");
        }
        if n.args.len() > PARAMETERS.len() {
            panic!("np.linalg.norm takes at most 4 arguments (x, ord, axis, keepdims), got {}", n.args.len());
        }
        for key in n.kwargs.keys() {
            if !PARAMETERS.contains(&key.as_str()) {
                panic!("np.linalg.norm: unexpected keyword argument `{key}`");
            }
        }
        let argument = |position: usize| -> Option<&ASTNode> {
            let name = PARAMETERS[position];
            match (n.args.get(position), n.kwargs.get(name)) {
                (Some(_), Some(_)) => panic!("np.linalg.norm got multiple values for argument '{name}'"),
                (Some(node), None) | (None, Some(node)) => Some(node),
                (None, None) => None,
            }
        };
        let x_node = argument(0).unwrap_or_else(|| panic!("np.linalg.norm: missing argument 'x'"));
        let (ord_node, axis_node, keepdims_node) = (argument(1), argument(2), argument(3));

        let x = self.visit(x_node);
        let ord = match ord_node {
            Some(node) => self.norm_order(node),
            None => NormOrder::Default,
        };
        let axis = match axis_node {
            Some(node) => {
                let value = self.visit(node);
                norm_axes(&value)
            }
            None => None,
        };
        let keepdims = match keepdims_node {
            Some(node) => {
                let value = self.visit(node);
                value
                    .bool_val()
                    .or_else(|| value.int_val().map(|v| v != 0))
                    .unwrap_or_else(|| panic!("np.linalg.norm: keepdims must be a compile-time True or False"))
            }
            None => false,
        };
        crate::ops::static_ndarray_ops::np_linalg_norm(&mut self.builder, NormCall { x, ord, axis, keepdims })
    }

    /// The compile-time `ord` of a norm call.
    fn norm_order(&mut self, node: &ASTNode) -> NormOrder {
        if let Some(positive) = infinity_sign(node) {
            return if positive { NormOrder::Inf } else { NormOrder::NegInf };
        }
        let value = self.visit(node);
        match &value {
            Value::None => NormOrder::Default,
            Value::String(s) if s.val == "fro" => NormOrder::Frobenius,
            Value::String(s) if s.val == "nuc" => NormOrder::Nuclear,
            Value::String(s) => panic!("np.linalg.norm: Invalid norm order '{}'", s.val),
            _ => match value.int_val().map(|v| v as f64).or_else(|| value.float_val()) {
                Some(p) => NormOrder::Number(p),
                None => panic!("np.linalg.norm: ord must be a compile-time constant"),
            },
        }
    }
}

/// `Some(true)` for `np.inf` / `math.inf` / `float('inf')` (also `np.Inf`,
/// `np.PINF`), `Some(false)` for their negation, `None` for anything else.
fn infinity_sign(node: &ASTNode) -> Option<bool> {
    match node {
        ASTNode::ASTUnaryOperator(u) if u.operator == "usub" => infinity_sign(&u.operand).map(|s| !s),
        ASTNode::ASTUnaryOperator(u) if u.operator == "uadd" => infinity_sign(&u.operand),
        ASTNode::ASTNamedAttribute(attr) => {
            let named_infinity = match (attr.target.as_deref(), attr.member.as_str()) {
                (Some("np"), "inf" | "Inf" | "infty" | "PINF") | (Some("math"), "inf") => attr.args.is_empty(),
                (None, "float") => matches!(
                    attr.args.as_slice(),
                    [ASTNode::ASTConstantString(s)] if matches!(s.value.to_ascii_lowercase().as_str(), "inf" | "+inf" | "infinity")
                ),
                _ => false,
            };
            named_infinity.then_some(true)
        }
        _ => None,
    }
}

/// `axis` as a list of axes: an int, or a tuple/list of ints.
fn norm_axes(value: &Value) -> Option<Vec<i64>> {
    let constant = |v: &Value| v.int_val().unwrap_or_else(|| panic!("np.linalg.norm: axis must be a compile-time integer or tuple of integers"));
    match value {
        Value::None => None,
        Value::Tuple(d) | Value::List(d) => Some(d.values.iter().map(constant).collect()),
        other => Some(vec![constant(other)]),
    }
}
