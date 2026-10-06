//! Whether two scalar cells hold the same value by construction.
//!
//! A program often writes one expression twice, and each occurrence builds
//! its own statements: in `x[x > c] = x[x > c] - d` the mask `x > c` is
//! computed once for the read and once for the write. Lowerings that pair
//! two operands by identity (the boolean-mask round trip in
//! `helpers::static_array::mask`) then miss the pairing. [`IRBuilder::
//! same_value`] recognizes such duplicates with the equivalence the
//! optimizer's duplicate elimination uses (`optim::duplicate_code_elimination`):
//! equal pure operations (the same `IR`, its data included, e.g. a constant's
//! value or an input's path) on operands that are the same values in turn.

use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRBuilder;

/// How many levels of operands are compared before giving up (a mask such as
/// `x > L / 2` needs three: the comparison, the division, the read).
const MAX_DEPTH: usize = 8;

impl IRBuilder {
    /// Whether the scalar cells `a` and `b` are the same value: the same
    /// statement, or statements that compute the same value by construction
    /// (see the module documentation). `false` when unsure (a cell without a
    /// statement, a memory read, any effectful or order-dependent statement,
    /// operands deeper than [`MAX_DEPTH`]), so a caller may only use a `true`.
    pub fn same_value(&self, a: &Value, b: &Value) -> bool {
        match (a.stmt_id(), b.stmt_id()) {
            (Some(x), Some(y)) => self.same_statement_value(x, y, MAX_DEPTH),
            _ => false,
        }
    }

    fn same_statement_value(&self, x: StmtId, y: StmtId, depth: usize) -> bool {
        if x == y {
            return true;
        }
        let (Some(sx), Some(sy)) = (self.stmts.get(x as usize), self.stmts.get(y as usize)) else {
            return false;
        };
        depth > 0
            && is_pure(&sx.ir)
            && sx.ir == sy.ir
            && sx.arguments.len() == sy.arguments.len()
            && sx.arguments.iter().zip(&sy.arguments).all(|(a, b)| self.same_statement_value(*a, *b, depth - 1))
    }
}

/// A statement whose value is a function of its operands and its own data:
/// not a memory access (its value depends on the writes before it) and not
/// effectful or order-sensitive (`IR::is_fixed`). An input read is pure (the
/// same path is the same value).
fn is_pure(ir: &IR) -> bool {
    matches!(ir, IR::ReadInteger { .. } | IR::ReadFloat { .. }) || !ir.is_fixed()
}
