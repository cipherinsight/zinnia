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

use crate::ast::DebugInfo;
use crate::error::ZinniaError;
use crate::types::{StmtId, Value};

// ---------------------------------------------------------------------------
// Resolver trait + StaticOnlyResolver default
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

    /// Cache-invalidation hook called by IR-mutating optim passes.
    ///
    /// `affected` is a (possibly empty) slice of mutated stmt ids. An empty
    /// slice is the conservative "everything possibly mutated" signal —
    /// P0 uses this default everywhere; P5 may refine to precise ids.
    fn on_ir_mutated(&mut self, _affected: &[StmtId]) {}
}

/// The default `Resolver`: no-op cache, delegates straight to the existing
/// `Value` accessors. Behaviourally identical to pre-P0 code.
#[derive(Debug, Default)]
pub struct StaticOnlyResolver;

impl StaticOnlyResolver {
    pub fn new() -> Self {
        Self
    }
}

impl Resolver for StaticOnlyResolver {
    fn resolve_int(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }

    fn resolve_bool(&mut self, val: &Value) -> Option<bool> {
        val.bool_val()
    }

    fn resolve_max(&mut self, val: &Value) -> Option<i64> {
        // For a fully-static integer, the value is its own max. P2's
        // `RangeResolver` will tighten this for non-literal expressions.
        val.int_val()
    }

    fn resolve_min(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }
}

// ---------------------------------------------------------------------------
// StaticInt wrapper + SiteKind + require_static_int
// ---------------------------------------------------------------------------

/// A required-constant integer at a specific call site.
///
/// Returned by [`require_static_int`] — i.e., obtained only after the
/// resolver has actually proved the wire is a compile-time constant.
///
/// Ergonomics convention (P0 decision): this type implements
/// `From<StaticInt> for i64`, so consumers can spell unwrapping as
/// `let n: i64 = static_n.into();` without `.0` ceremony.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticInt(pub i64);

impl StaticInt {
    /// Borrow the wrapped i64. Use sparingly — prefer `into()` /
    /// `From<StaticInt>` at consumer sites.
    pub fn get(self) -> i64 {
        self.0
    }
}

impl From<StaticInt> for i64 {
    fn from(s: StaticInt) -> i64 {
        s.0
    }
}

impl From<StaticInt> for u32 {
    fn from(s: StaticInt) -> u32 {
        s.0 as u32
    }
}

impl From<StaticInt> for usize {
    fn from(s: StaticInt) -> usize {
        s.0 as usize
    }
}

/// Where a "must-be-constant" integer requirement is being enforced.
///
/// Exists primarily to centralise diagnostic-message formatting so every
/// site speaks the same "<thing> must be a compile-time constant int"
/// dialect. Add a new variant when migrating a site whose rejection
/// message doesn't already fit.
#[derive(Clone, Copy, Debug)]
pub enum SiteKind {
    /// A shape element of an array constructor, at the given axis.
    ShapeAxis(usize),
    /// `start` argument of a slice expression.
    SliceStart,
    /// `stop` argument of a slice expression.
    SliceStop,
    /// `step` argument of a slice expression.
    SliceStep,
    /// `start` argument of `range(...)`.
    RangeStart,
    /// `stop` argument of `range(...)`.
    RangeStop,
    /// `step` argument of `range(...)`.
    RangeStep,
    /// A target dimension of `arr.reshape(...)` / `np.reshape(arr, ...)`.
    ReshapeDim,
    /// `repeats` argument of `np.repeat(arr, k)`.
    RepeatCount,
    /// `n` argument of `np.split(arr, n)`.
    SplitSections,
    /// An `axis=` argument (sum, transpose, swapaxes, …).
    Axis,
    /// Position passed to `np.expand_dims` / newaxis.
    NewAxisPosition,
    /// `num` argument of `np.linspace`.
    LinspaceNum,
    /// Anything not yet enumerated. The string is a short site label;
    /// promote it to a named variant if it recurs.
    Other(&'static str),
}

impl SiteKind {
    /// Render a human-readable diagnostic for a "this must be a compile-time
    /// constant int" rejection at this site. One source of truth for the
    /// wording.
    pub fn diagnostic(&self) -> String {
        match self {
            SiteKind::ShapeAxis(i) => format!(
                "shape element at axis {} must be a compile-time constant int",
                i
            ),
            SiteKind::SliceStart => {
                "slice start must be a compile-time constant int".to_string()
            }
            SiteKind::SliceStop => {
                "slice stop must be a compile-time constant int".to_string()
            }
            SiteKind::SliceStep => {
                "slice step must be a compile-time constant int".to_string()
            }
            SiteKind::RangeStart => {
                "range start must be a compile-time constant int".to_string()
            }
            SiteKind::RangeStop => {
                "range stop must be a compile-time constant int".to_string()
            }
            SiteKind::RangeStep => {
                "range step must be a compile-time constant int".to_string()
            }
            SiteKind::ReshapeDim => {
                "reshape target dimension must be a compile-time constant int"
                    .to_string()
            }
            SiteKind::RepeatCount => {
                "repeat count must be a compile-time constant int".to_string()
            }
            SiteKind::SplitSections => {
                "split sections must be a compile-time constant int".to_string()
            }
            SiteKind::Axis => {
                "axis argument must be a compile-time constant int".to_string()
            }
            SiteKind::NewAxisPosition => {
                "new-axis position must be a compile-time constant int".to_string()
            }
            SiteKind::LinspaceNum => {
                "linspace `num` must be a compile-time constant int".to_string()
            }
            SiteKind::Other(label) => {
                format!("{} must be a compile-time constant int", label)
            }
        }
    }
}

/// Format the diagnostic for a `require_static_*` failure, optionally
/// prefixed with debug-info location.
fn format_diagnostic(site: SiteKind, dbg: Option<&DebugInfo>) -> String {
    let base = site.diagnostic();
    match dbg {
        Some(d) => match (d.line, d.col) {
            (Some(line), Some(col)) => format!("{} (at line {}, col {})", base, line, col),
            (Some(line), None) => format!("{} (at line {})", base, line),
            _ => base,
        },
        None => base,
    }
}

/// The single chokepoint for "this integer site must be a compile-time
/// constant" requirements.
///
/// Routes the query through the IRBuilder's resolver (today:
/// [`StaticOnlyResolver`], tomorrow: SMT/range-augmented). Returns a typed
/// [`StaticInt`] on success, or a [`ZinniaError`] with a uniform diagnostic
/// referring to the [`SiteKind`].
///
/// Per-category migration of existing `int_val().expect("constant")` sites
/// onto this API is staged across follow-up commits in the SMT epic.
pub fn require_static_int(
    b: &mut crate::builder::IRBuilder,
    val: &Value,
    site: SiteKind,
    dbg: Option<&DebugInfo>,
) -> Result<StaticInt, ZinniaError> {
    match b.resolver_mut().resolve_int(val) {
        Some(n) => Ok(StaticInt(n)),
        None => Err(ZinniaError {
            message: format_diagnostic(site, dbg),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ScalarValue;

    #[test]
    fn static_only_resolver_matches_int_val() {
        let mut r = StaticOnlyResolver::new();
        let v = Value::Integer(ScalarValue::constant(42));
        assert_eq!(r.resolve_int(&v), Some(42));
        assert_eq!(r.resolve_max(&v), Some(42));
        assert_eq!(r.resolve_min(&v), Some(42));
    }

    #[test]
    fn static_only_resolver_unknown_for_runtime_int() {
        let mut r = StaticOnlyResolver::new();
        let v = Value::Integer(ScalarValue::runtime(0));
        assert_eq!(r.resolve_int(&v), None);
    }

    #[test]
    fn static_int_into_i64() {
        let s = StaticInt(7);
        let n: i64 = s.into();
        assert_eq!(n, 7);
    }

    #[test]
    fn site_kind_diagnostic_includes_axis() {
        assert!(SiteKind::ShapeAxis(3)
            .diagnostic()
            .contains("axis 3"));
    }
}
