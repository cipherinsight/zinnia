//! Every input element is checked to lie in its dtype's domain.
//!
//! A circuit input is a field element chosen by the prover. Without a check,
//! a proof could rest on a value no NumPy input has (an integer beyond int64,
//! a "boolean" equal to 2, a float whose fixed-point code wraps), and the
//! statement "these outputs come from some valid input" would not hold. So
//! each input element is constrained once, when it is read:
//!
//! * `Integer`: `i64::MIN <= x <= i64::MAX`, the signed 64-bit window;
//! * `Float` (Q32 fixed point, code `round(v * 2^32)`): `-2^31 <= v <=
//!   2^31 - 2^-22`. The lower bound is the window's bottom code `-2^63`; the
//!   upper bound is the largest double below `2^31`, whose code `2^63 - 2^10`
//!   is the largest code any float64 in the window quantizes to (doubles in
//!   `[2^30, 2^31)` are `2^-22` apart). `2^31` itself (code `2^63`) is not a
//!   constant the backends can hold;
//! * `Boolean`: `x * (x - 1) == 0` (`AssertBit`).
//!
//! Soundness of the comparisons: each is an exact signed comparison on every
//! backend for operands of small magnitude and fails (or has no witness) for
//! the others: the mock compares by the sign of `x - c` over the field, and
//! the two bounds together leave exactly the window (a field element above it
//! makes `MAX - x` negative, one below makes `x - MIN` negative); halo2-base
//! puts both operands of a signed comparison in the signed 64-bit window;
//! Noir compares through signed projections. Cost: the backends that track
//! proven ranges decide the second comparison from the first (and later
//! comparisons on the input skip their own window check).
//!
//! Nothing is elided where the input matters: no check that exists when an
//! input is read is equal to or stronger than these. An input that nothing
//! but its own check reads after optimization (e.g. a parameter the kernel
//! never uses) does not take part in the proven statement: any valid value
//! satisfies the check and the outputs do not depend on it. Its check is
//! dropped after optimization ([`drop_checks_of_unread_inputs`]), so the
//! read is removed as before.

use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value, ZinniaType};

use super::IRGenerator;

/// The largest double below `2^31`: the largest Q32 float input.
const FLOAT_INPUT_MAX: f64 = 2147483648.0 - 1.0 / 4194304.0;
/// `-2^31`: the smallest Q32 float input (code `-2^63`).
const FLOAT_INPUT_MIN: f64 = -2147483648.0;

impl IRGenerator {
    /// Constrain a freshly read input element to its dtype's domain.
    pub(crate) fn check_input_domain(&mut self, value: &Value, dtype: &ZinniaType) {
        match dtype {
            ZinniaType::Boolean => {
                self.builder.ir_assert_bit(value);
            }
            ZinniaType::Integer => {
                let lo = self.builder.ir_constant_int(i64::MIN);
                let hi = self.builder.ir_constant_int(i64::MAX);
                let above = self.builder.ir_greater_than_or_equal_i(value, &lo);
                self.builder.ir_assert(&above);
                let below = self.builder.ir_less_than_or_equal_i(value, &hi);
                self.builder.ir_assert(&below);
            }
            ZinniaType::Float => {
                let lo = self.builder.ir_constant_float(FLOAT_INPUT_MIN);
                let hi = self.builder.ir_constant_float(FLOAT_INPUT_MAX);
                let above = self.builder.ir_greater_than_or_equal_f(value, &lo);
                self.builder.ir_assert(&above);
                let below = self.builder.ir_less_than_or_equal_f(value, &hi);
                self.builder.ir_assert(&below);
            }
            _ => {}
        }
    }
}

/// Whether statement `cmp` is a domain-check comparison of `read` (the read
/// on the left, the matching window constant on the right).
fn is_domain_comparison(graph: &IRGraph, cmp: StmtId, read: StmtId) -> bool {
    let stmt = &graph.stmts[cmp as usize];
    let [lhs, rhs] = stmt.arguments[..] else { return false };
    if lhs != read {
        return false;
    }
    match (&stmt.ir, &graph.stmts[rhs as usize].ir) {
        (IR::GteI, IR::ConstantInt { value }) => *value == i64::MIN,
        (IR::LteI, IR::ConstantInt { value }) => *value == i64::MAX,
        (IR::GteF, IR::ConstantFloat { value }) => *value == FLOAT_INPUT_MIN,
        (IR::LteF, IR::ConstantFloat { value }) => *value == FLOAT_INPUT_MAX,
        _ => false,
    }
}

/// The check statements to drop for a private input `read` whose only
/// consumers are its domain checks: the comparisons, their `Assert`s and an
/// `AssertBit`. `None` when anything else reads the input.
fn unread_input_checks(graph: &IRGraph, read: StmtId) -> Option<Vec<StmtId>> {
    let mut checks = Vec::new();
    for &consumer in &graph.out_links[read as usize] {
        match graph.stmts[consumer as usize].ir {
            IR::AssertBit => checks.push(consumer),
            _ if is_domain_comparison(graph, consumer, read) => {
                let asserts = &graph.out_links[consumer as usize];
                if !asserts.iter().all(|&a| matches!(graph.stmts[a as usize].ir, IR::Assert)) {
                    return None;
                }
                checks.push(consumer);
                checks.extend(asserts.iter().copied());
            }
            _ => return None,
        }
    }
    Some(checks)
}

/// Remove the domain checks of private inputs that nothing else reads (see
/// the module docs); a dead-code pass afterwards removes the reads.
pub fn drop_checks_of_unread_inputs(mut graph: IRGraph) -> IRGraph {
    let mut dropped: Vec<StmtId> = Vec::new();
    for (index, stmt) in graph.stmts.iter().enumerate() {
        let private_read = matches!(
            stmt.ir,
            IR::ReadInteger { is_public: false, .. } | IR::ReadFloat { is_public: false, .. }
        );
        if private_read {
            if let Some(checks) = unread_input_checks(&graph, index as StmtId) {
                dropped.extend(checks);
            }
        }
    }
    if !dropped.is_empty() {
        dropped.sort_unstable();
        dropped.dedup();
        graph.remove_stmt_bunch(&dropped);
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_bounds_are_the_extreme_q32_codes_of_doubles() {
        let code = |v: f64| crate::prove::field::quantize(v, 32);
        assert_eq!(code(FLOAT_INPUT_MIN), -(1i128 << 63));
        assert_eq!(code(FLOAT_INPUT_MAX), (1i128 << 63) - (1 << 10));
        // The next double up is 2^31, whose code leaves the window.
        let next = f64::from_bits(FLOAT_INPUT_MAX.to_bits() + 1);
        assert_eq!(code(next), 1i128 << 63);
    }
}
