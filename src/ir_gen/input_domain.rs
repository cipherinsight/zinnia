//! Every input element is checked to lie in its dtype's domain.
//!
//! A circuit input is a field element chosen by the prover. Without a check,
//! a proof could rest on a value no NumPy input has (an integer beyond int64,
//! a "boolean" equal to 2, a float whose fixed-point code wraps), and the
//! statement "these outputs come from some valid input" would not hold. So
//! each input element is constrained once, when it is read:
//!
//! * `Integer`: `i64::MIN <= x <= i64::MAX`, the signed 64-bit window;
//! * `Float` (Q32 fixed point, code `round(v * 2^32)`): the code in the same
//!   window, `-2^63 <= code <= 2^63 - 1` (`-2^31 <= v < 2^31`). The code is
//!   compared with the integer bounds: every backend compares Q32 floats as
//!   their integer codes, and the top code `2^63 - 1` has no float constant
//!   (`(2^63 - 1) / 2^32` rounds to `2^31`, whose code leaves the window);
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

impl IRGenerator {
    /// Constrain a freshly read input element to its dtype's domain.
    pub(crate) fn check_input_domain(&mut self, value: &Value, dtype: &ZinniaType) {
        match dtype {
            ZinniaType::Boolean => {
                self.builder.ir_assert_bit(value);
            }
            // A float's Q32 code is checked like an integer (module docs).
            ZinniaType::Integer | ZinniaType::Float => {
                let lo = self.builder.ir_constant_int(i64::MIN);
                let hi = self.builder.ir_constant_int(i64::MAX);
                let above = self.builder.ir_greater_than_or_equal_i(value, &lo);
                self.builder.ir_assert(&above);
                let below = self.builder.ir_less_than_or_equal_i(value, &hi);
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
