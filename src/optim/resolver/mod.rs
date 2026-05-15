//! P0 — `Resolver` seam for the SMT-resolver epic.
//!
//! This module defines the architectural seams introduced by phase 0 of
//! `compiler.epic-restore-smt-reasoning`:
//!
//! * The [`Resolver`] trait — the abstraction every "is this value provably
//!   constant?" query goes through. P0 ships [`StaticOnlyResolver`], whose
//!   behaviour is byte-for-byte identical to today (it just delegates to the
//!   existing `Value::int_val` / `bool_val` accessors). P1 adds an
//!   `SmtResolver`, P2 adds a `RangeResolver`, and call sites stay unchanged.
//!
//! * The [`StaticInt`] typed wrapper plus the [`SiteKind`] enum and the
//!   [`require_static_int`] helper. These give us one chokepoint for
//!   "must-be-compile-time-constant" integer call sites with informative,
//!   uniform diagnostics. The bulk of the migration of the
//!   `int_val().expect("constant")` sites onto this API is staged across
//!   per-category follow-up commits.
//!
//! No behaviour change in P0. The default resolver matches today's semantics.
//!
//! The implementation is split across focused submodules; this file owns
//! the `Resolver` trait, the kill-switch helper, and re-exports the
//! public surface.

use crate::ir::IRStatement;
use crate::types::{StmtId, Value};

mod chokepoints;
mod discharge;
mod intervals;
mod layered;
mod relays;
mod smt;
mod static_only;

pub use chokepoints::{
    require_provable_static_int, require_static_int, require_static_or_bounded_int,
    resolve_int_or_bounded, BoundedInt, SiteKind, StaticInt,
};
pub use discharge::{discharge_index_in_range, discharge_slice_bound, probe_in_range};
pub use intervals::{
    interval_fact_for_float_binary, interval_fact_for_int_binary,
};
pub(crate) use intervals::derive_bounds_from_facts;
pub use layered::LayeredResolver;
pub use relays::{
    relay_arccos_output_interval, relay_exp_output_interval,
    relay_forall_eq_const_from_all_inputs, relay_forall_eq_const_from_input,
    relay_log_output_interval, relay_reduction_output_interval_int,
    relay_sqrt_output_interval,
};
pub use smt::SmtResolver;
pub use static_only::StaticOnlyResolver;

/// `ZINNIA_REQ_DISABLE=1` is the A/B-harness kill switch for the R/E/Q
/// machinery: when set, `prove(_)` short-circuits to `Unknown`, the
/// strategy dispatcher always runs the default lowering, and the
/// `relay_*` helpers no-op. Mirrors the read pattern of
/// `op_requires_strict()` in `builder.rs` — a single env lookup per
/// call, so toggling from a test or harness takes effect immediately.
///
/// Soundness invariant: when `prove` is forced to Unknown, the
/// `discharge_requires` lenient branch still emits the witness check
/// (`IR::Assert`), so preconditions are enforced at proof time. The
/// kill switch must never bypass that floor — see the
/// `disable_switch_preserves_witness_emit` test.
pub fn req_disabled() -> bool {
    std::env::var("ZINNIA_REQ_DISABLE")
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Resolver trait
// ---------------------------------------------------------------------------

/// The "is this value provably constant?" abstraction.
///
/// The `&mut self` receiver is intentional: P1's `SmtResolver` will memoise
/// per-ptr query results, and P2's `RangeResolver` will accumulate interval
/// information across queries. P0's [`StaticOnlyResolver`] is stateless, so
/// the receiver is unused there — but holding the line on `&mut self` now
/// avoids a churning API change in P1.
///
/// `Send + Sync` are required because [`crate::ir::IRGraph`] is held by a
/// `#[pyclass]` (`CompiledIR`), which requires its fields to be thread-safe.
/// Concrete impls must respect this bound (e.g., a future SMT context held
/// inside a resolver needs to be guarded with a mutex if it isn't already
/// thread-safe).
pub trait Resolver: Send + Sync {
    /// Resolve `val` to a compile-time integer if provably constant.
    fn resolve_int(&mut self, val: &Value) -> Option<i64>;

    /// Resolve `val` to a compile-time boolean if provably constant.
    fn resolve_bool(&mut self, val: &Value) -> Option<bool>;

    /// Upper bound on `val` if provable.
    ///
    /// For [`StaticOnlyResolver`] this collapses to the literal itself
    /// (max of a constant is the constant). P2's range resolver gives a
    /// tighter bound, P1's SMT resolver discharges via maximisation.
    fn resolve_max(&mut self, val: &Value) -> Option<i64>;

    /// Lower bound on `val` if provable.
    fn resolve_min(&mut self, val: &Value) -> Option<i64>;

    /// Resolve `val` to a compile-time integer, with the IR statement vector
    /// available for traversal. P1's `SmtResolver` overrides this and walks
    /// the dependency graph; resolvers that don't care about the IR (i.e.
    /// [`StaticOnlyResolver`]) just delegate to the static-val variant.
    ///
    /// Why a separate method instead of putting `&[IRStatement]` on every
    /// `resolve_*`: the trait is invoked through `IRBuilder::resolver_mut()`,
    /// which currently exposes `&mut dyn Resolver` on its own. The
    /// `_with_stmts` variants are wired through a dedicated chokepoint
    /// (`IRBuilder::split_resolver_and_stmts`) so the borrow-checker can
    /// hand out `&mut resolver` and `&[IRStatement]` simultaneously without
    /// a churning API change at every call site.
    fn resolve_int_with_stmts(
        &mut self,
        val: &Value,
        _stmts: &[IRStatement],
    ) -> Option<i64> {
        self.resolve_int(val)
    }

    fn resolve_bool_with_stmts(
        &mut self,
        val: &Value,
        _stmts: &[IRStatement],
    ) -> Option<bool> {
        self.resolve_bool(val)
    }

    fn resolve_max_with_stmts(
        &mut self,
        val: &Value,
        _stmts: &[IRStatement],
    ) -> Option<i64> {
        self.resolve_max(val)
    }

    fn resolve_min_with_stmts(
        &mut self,
        val: &Value,
        _stmts: &[IRStatement],
    ) -> Option<i64> {
        self.resolve_min(val)
    }

    /// Cache-invalidation hook called by IR-mutating optim passes.
    ///
    /// `affected` is a (possibly empty) slice of mutated stmt ids. An empty
    /// slice is the conservative "everything possibly mutated" signal —
    /// P0 uses this default everywhere; P5 may refine to precise ids.
    fn on_ir_mutated(&mut self, _affected: &[StmtId]) {}

    /// P5: expose the SMT-pipeline telemetry handle, when the resolver
    /// has one. Layered/Range/Smt resolvers return Some; StaticOnly
    /// returns None. The handle is `Arc`'d, so callers can hold it past
    /// the resolver's lifetime to print a summary at end of compilation.
    fn telemetry_handle(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        None
    }
}

#[cfg(test)]
mod tests;
