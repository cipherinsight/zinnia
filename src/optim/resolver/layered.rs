//! LayeredResolver — composition of resolvers, cheap-first dispatch.
//!
//! P2 — the layered-resolver pattern from the epic spec (design principle 1:
//! "cheap analyses first, SMT as backstop"). Holds a list of `Box<dyn Resolver>`
//! layers; on each query, walks them in order and returns the first
//! `Some(_)` answer.
//!
//! Typical composition for the full epic: `range → static → SMT`. P2 ships
//! the construction and a `range_then_smt` helper. Wiring it in as the
//! default `IRBuilder` / `IRGraph` resolver is P3.
//!
//! `on_ir_mutated` fans out to every layer so each one's invalidation policy
//! fires.

use crate::ir::IRStatement;
use crate::types::{StmtId, Value};

use super::smt::SmtResolver;
use super::Resolver;

/// A pipeline of resolvers, queried in order. First `Some(_)` wins.
pub struct LayeredResolver {
    layers: Vec<Box<dyn Resolver>>,
    /// P5 telemetry, accumulating cross-layer counters. Constructed by
    /// `range_then_smt` (so the range and SMT layers share one). Public
    /// constructors `new` / `new_with_telemetry` let callers wire it
    /// however they like.
    telemetry: std::sync::Arc<crate::optim::telemetry::SmtTelemetry>,
}

impl std::fmt::Debug for LayeredResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredResolver")
            .field("num_layers", &self.layers.len())
            .finish()
    }
}

impl LayeredResolver {
    /// Build a pipeline from an explicit list of layers. The telemetry
    /// stays disconnected from the sub-layers (each layer keeps its own).
    /// For shared telemetry across layers, use `range_then_smt` or
    /// `new_with_telemetry`.
    pub fn new(layers: Vec<Box<dyn Resolver>>) -> Self {
        Self {
            layers,
            telemetry: crate::optim::telemetry::SmtTelemetry::new(),
        }
    }

    /// Build a pipeline with an explicit shared telemetry. Caller is
    /// responsible for wiring the same telemetry into each sub-layer (via
    /// `RangeResolver::with_telemetry` / `SmtResolver::with_telemetry`).
    pub fn new_with_telemetry(
        layers: Vec<Box<dyn Resolver>>,
        telemetry: std::sync::Arc<crate::optim::telemetry::SmtTelemetry>,
    ) -> Self {
        Self { layers, telemetry }
    }

    /// The canonical P2 composition: `RangeResolver → SmtResolver`.
    /// Intended consumer for P3+. Range handles the bounded-loop-index /
    /// modular / mask cases; SMT handles symbolic relations range can't see
    /// (e.g., `select(x == 5, 100, 100)` where the cond depends on a free
    /// variable).
    ///
    /// Both sub-layers share one telemetry handle so the end-of-compilation
    /// summary covers the whole pipeline.
    pub fn range_then_smt() -> Self {
        Self::range_then_smt_with_timeout(500)
    }

    /// Same as `range_then_smt` but with an explicit Z3 per-query timeout
    /// (ms). P5 uses this so callers can tighten the budget without first
    /// constructing the SMT layer manually.
    pub fn range_then_smt_with_timeout(timeout_ms: u64) -> Self {
        Self::range_then_smt_with_budget(timeout_ms, usize::MAX)
    }

    /// Same as `range_then_smt_with_timeout` but with an additional cap on
    /// the per-query formula size (number of IR statements visited by the
    /// reverse-reachability walk). Beyond this cap the walk aborts and the
    /// SmtResolver returns None — counted as `queries_skipped_oversized`
    /// in telemetry. P5 commit 3 uses this so callers can configure a
    /// pragmatic budget that bounds the worst-case query without changing
    /// the timeout.
    pub fn range_then_smt_with_budget(
        timeout_ms: u64,
        max_formula_size: usize,
    ) -> Self {
        let telemetry = crate::optim::telemetry::SmtTelemetry::new();
        let range = crate::optim::range::RangeResolver::new()
            .with_telemetry(std::sync::Arc::clone(&telemetry));
        let smt = SmtResolver::new()
            .with_timeout(timeout_ms)
            .with_max_formula_size(max_formula_size)
            .with_telemetry(std::sync::Arc::clone(&telemetry));
        Self::new_with_telemetry(
            vec![Box::new(range), Box::new(smt)],
            telemetry,
        )
    }

    /// Borrow the shared telemetry handle. Used by the compile entry-point
    /// to surface the summary to stderr at end of compilation.
    pub fn telemetry(&self) -> std::sync::Arc<crate::optim::telemetry::SmtTelemetry> {
        std::sync::Arc::clone(&self.telemetry)
    }
}

impl Resolver for LayeredResolver {
    fn resolve_int(&mut self, val: &Value) -> Option<i64> {
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_int(val) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_bool(&mut self, val: &Value) -> Option<bool> {
        for layer in self.layers.iter_mut() {
            if let Some(b) = layer.resolve_bool(val) {
                return Some(b);
            }
        }
        None
    }

    fn resolve_max(&mut self, val: &Value) -> Option<i64> {
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_max(val) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_min(&mut self, val: &Value) -> Option<i64> {
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_min(val) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_int_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        use std::sync::atomic::Ordering;
        self.telemetry.queries_total.fetch_add(1, Ordering::Relaxed);
        if val.int_val().is_some() {
            self.telemetry
                .queries_static_val_hit
                .fetch_add(1, Ordering::Relaxed);
        }
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_int_with_stmts(val, stmts) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_bool_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<bool> {
        use std::sync::atomic::Ordering;
        self.telemetry.queries_total.fetch_add(1, Ordering::Relaxed);
        if val.bool_val().is_some() {
            self.telemetry
                .queries_static_val_hit
                .fetch_add(1, Ordering::Relaxed);
        }
        for layer in self.layers.iter_mut() {
            if let Some(b) = layer.resolve_bool_with_stmts(val, stmts) {
                return Some(b);
            }
        }
        None
    }

    fn resolve_max_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        use std::sync::atomic::Ordering;
        self.telemetry.queries_total.fetch_add(1, Ordering::Relaxed);
        if val.int_val().is_some() {
            self.telemetry
                .queries_static_val_hit
                .fetch_add(1, Ordering::Relaxed);
        }
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_max_with_stmts(val, stmts) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_min_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        use std::sync::atomic::Ordering;
        self.telemetry.queries_total.fetch_add(1, Ordering::Relaxed);
        if val.int_val().is_some() {
            self.telemetry
                .queries_static_val_hit
                .fetch_add(1, Ordering::Relaxed);
        }
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_min_with_stmts(val, stmts) {
                return Some(n);
            }
        }
        None
    }

    fn on_ir_mutated(&mut self, affected: &[StmtId]) {
        for layer in self.layers.iter_mut() {
            layer.on_ir_mutated(affected);
        }
    }

    fn telemetry_handle(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        Some(std::sync::Arc::clone(&self.telemetry))
    }
}
