//! IR Generator — visitor pattern over Zinnia AST, produces IRGraph.
//! Ports `zinnia/compile/ir/ir_gen.py` (854 lines).

use std::collections::HashMap;

use crate::ast::*;
use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_ctx::IRContext;
use crate::optim::resolver::{LayeredResolver, Resolver, StaticOnlyResolver};
use crate::types::Value;

mod visitors;
mod named_attr;
mod list_methods;
mod chip_external;
mod helpers;

#[cfg(test)]
mod tests;

/// Check if an assignment target node has the `star` flag set.
pub(super) fn is_starred_target(node: &ASTNode) -> bool {
    match node {
        ASTNode::ASTNameAssignTarget(t) => t.star,
        ASTNode::ASTSubscriptAssignTarget(t) => t.star,
        ASTNode::ASTStarredExpr(_) => true,
        _ => false,
    }
}

// Re-export SliceIndex from types for backward compatibility
pub use crate::types::SliceIndex;

/// Configuration for the IR generator.
///
/// Note: this is the natural Rust-side home for compile-time knobs. The
/// `ZinniaConfig` struct found via `grep ZinniaConfig` is the *halo2
/// circuit* config (column / gate layout) — an unrelated concept; not the
/// place for compile knobs. The kanban spec's `ZinniaConfig::smt_*` knobs
/// land here because `IRGenConfig` is what `lib.rs::compile_circuit`
/// already plumbs through from the JSON input on the Python side.
#[derive(Debug, Clone)]
pub struct IRGenConfig {
    pub loop_limit: u32,
    pub recursion_limit: u32,
    /// SMT-resolver toggle. When `true`, `IRGenerator::new` installs
    /// [`crate::optim::resolver::LayeredResolver::range_then_smt`]; when
    /// `false`, [`crate::optim::resolver::StaticOnlyResolver`] (the pre-P3
    /// behaviour). P3 wired the toggle but the **default is `false`**:
    /// the benchmark sweep at the time of the flip showed a `+1` net
    /// coverage gain (`grayscott` flips TIMEOUT→PASS) but **>5× compile-
    /// time slowdowns on 12 / 104 dual-pass benchmarks** (peak 66× on
    /// `guerre`, 25× on `perm`). Per the spec exit criterion ("all
    /// current tests pass + measurable coverage gain — don't sacrifice
    /// the first for the second") that is a halt condition.
    /// Flip back to `true` once the slowdown root cause is identified
    /// (likely SMT-layer cost on hot-path queries; see P5 telemetry).
    pub smt_enable: bool,
    /// Per-query Z3 timeout in milliseconds. Default 100 ms (P5
    /// commit 3: tightened from 500 ms — the AST→IR-phase telemetry
    /// across the whole benchmark suite shows zero SMT queries, but if
    /// future call sites do reach the SMT layer, a 100 ms ceiling caps
    /// the worst-case at < 2× of the suite's current quietest fail
    /// budget. Lower = faster compile, weaker reasoning. Higher =
    /// slower compile, stronger reasoning.
    pub smt_query_timeout_ms: u64,
    /// Maximum number of IR statements the reverse-reachability walk
    /// will visit per SMT query before aborting (returning `None` and
    /// counting the query as `queries_skipped_oversized` in telemetry).
    /// Default 4096 per spec. P5 commit 3.
    pub smt_max_formula_size: usize,
    /// P5 telemetry knob. When `true`, the compiler logs
    /// `SmtTelemetry::summary()` to stderr at end of compilation. Default
    /// `false`. Wire-only — the counters are always collected (their cost
    /// is a handful of `AtomicUsize::fetch_add`s per query); this flag
    /// only controls whether we print the summary.
    pub smt_log_telemetry: bool,
}

impl Default for IRGenConfig {
    fn default() -> Self {
        Self {
            loop_limit: 256,
            recursion_limit: 16,
            // See struct docs: held at `false` until the P3 compile-time
            // regression is resolved.
            smt_enable: false,
            smt_query_timeout_ms: 100,
            smt_max_formula_size: 4096,
            smt_log_telemetry: false,
        }
    }
}

/// Registered chip information for the IR generator.
#[derive(Debug, Clone)]
pub struct RegisteredChip {
    pub chip_ast: serde_json::Value,  // The ASTChip dict
    pub return_dt: serde_json::Value, // Return type descriptor
}

/// Registered external function information for the IR generator.
#[derive(Debug, Clone)]
pub struct RegisteredExternal {
    pub return_dt: serde_json::Value, // Return type descriptor
}

/// The IR Generator — walks the AST and produces an IRGraph.
pub struct IRGenerator {
    pub builder: IRBuilder,
    pub ctx: IRContext,
    config: IRGenConfig,
    registered_chips: HashMap<String, RegisteredChip>,
    registered_externals: HashMap<String, RegisteredExternal>,
    recursion_depth: u32,
    next_external_store_idx: u32,
    /// P4 round 2 — per-active-call snapshot of integer-arg bindings,
    /// keyed by chip name. When `visit_chip_call("foo", args, …)` runs
    /// and finds a prior frame with the same name, that frame is the
    /// parent of a recursive call: we diff `args[i]` against the
    /// snapshotted parent value to pick the recursion measure.
    /// `int_args[i]` is `None` when the i-th input parameter wasn't
    /// integer-valued (so the heuristic skips it).
    pub(crate) chip_call_stack: Vec<ChipCallFrame>,
}

/// One entry in [`IRGenerator::chip_call_stack`]. See round-2 doc on
/// `chip_call_stack` for semantics.
#[derive(Debug, Clone)]
pub(crate) struct ChipCallFrame {
    pub chip_name: String,
    /// One slot per chip-input parameter. `Some(n)` if the bound value
    /// was an integer with a known compile-time int_val; `None`
    /// otherwise (non-integer parameter, or symbolic int with no
    /// static_val cache hit). Used by the recursion-measure heuristic
    /// in `visit_chip_call`.
    pub int_args: Vec<Option<i64>>,
    /// Forward-looking depth allowance for this frame and any
    /// recursive-call descendants of it. Initialised at entry to
    /// `min(parent.remaining_bound - 1, freshly_resolved_bound)`,
    /// always `≤ recursion_limit`. A descendant recursive call panics
    /// if its parent's `remaining_bound == 0`. Non-recursive entries
    /// (no prior same-name frame) start at `recursion_limit` — today's
    /// behaviour, no tightening.
    pub remaining_bound: u32,
}

impl IRGenerator {
    pub fn new(config: IRGenConfig) -> Self {
        let mut builder = IRBuilder::new();
        // P3: flip the default resolver from `StaticOnlyResolver` to the
        // layered `range → SMT` pipeline when `cfg.smt_enable` is true.
        // Falls back to `StaticOnlyResolver` (today's pre-P3 behaviour)
        // when the flag is off — the safety net for diagnosing whether
        // SMT introduces a regression.
        let resolver: Box<dyn Resolver> = if config.smt_enable {
            Box::new(LayeredResolver::range_then_smt_with_budget(
                config.smt_query_timeout_ms,
                config.smt_max_formula_size,
            ))
        } else {
            Box::new(StaticOnlyResolver::new())
        };
        builder.set_resolver(resolver);
        Self {
            builder,
            ctx: IRContext::new(),
            config,
            registered_chips: HashMap::new(),
            registered_externals: HashMap::new(),
            recursion_depth: 0,
            next_external_store_idx: 1,
            chip_call_stack: Vec::new(),
        }
    }

    /// Main entry point: generate an IRGraph from an AST circuit.
    pub fn generate(mut self, ast: &ASTCircuit) -> IRGraph {
        self.visit_circuit(ast);
        let smt_enable = self.config.smt_enable;
        let timeout_ms = self.config.smt_query_timeout_ms;
        let max_formula_size = self.config.smt_max_formula_size;
        // Snapshot the AST→IR-phase telemetry before exporting the graph.
        // Per P1 option (b) each phase has its own resolver / cache, but
        // the telemetry handle the optim phase will use is a fresh one;
        // we surface the AST→IR snapshot so the end-of-compile summary
        // includes it.
        let astgen_telemetry = self.builder.resolver_telemetry();
        let mut graph = self.builder.export_ir_graph();
        // P3: also propagate the resolver choice onto the resulting
        // `IRGraph` so optim passes that consult the resolver (e.g.
        // through `IRGraph::split_resolver_and_stmts`) see the same
        // policy as the AST→IR phase. Per P1's option (b), each phase
        // owns its own resolver / cache; we just ensure the choice
        // matches the config.
        let resolver: Box<dyn Resolver> = if smt_enable {
            Box::new(LayeredResolver::range_then_smt_with_budget(
                timeout_ms,
                max_formula_size,
            ))
        } else {
            Box::new(StaticOnlyResolver::new())
        };
        graph.set_resolver(resolver);
        if let Some(t) = astgen_telemetry {
            graph.set_astgen_telemetry(t);
        }
        graph
    }

    /// Generate from a JSON string (the bridge entry point).
    pub fn generate_from_json(config: IRGenConfig, ast_json: &str) -> Result<IRGraph, String> {
        let node: ASTNode = serde_json::from_str(ast_json)
            .map_err(|e| format!("AST parse error: {}", e))?;
        match node {
            ASTNode::ASTCircuit(circuit) => {
                let gen = IRGenerator::new(config);
                Ok(gen.generate(&circuit))
            }
            _ => Err("Expected ASTCircuit at top level".to_string()),
        }
    }

    /// Generate from JSON with chips and externals.
    pub fn generate_from_json_with_chips(
        config: IRGenConfig,
        ast_json: &str,
        chips: &HashMap<String, serde_json::Value>,
        externals: &HashMap<String, serde_json::Value>,
    ) -> Result<IRGraph, String> {
        let node: ASTNode = serde_json::from_str(ast_json)
            .map_err(|e| format!("AST parse error: {}", e))?;
        match node {
            ASTNode::ASTCircuit(circuit) => {
                let mut gen = IRGenerator::new(config);
                // Register chips
                for (name, chip_data) in chips {
                    gen.registered_chips.insert(name.clone(), RegisteredChip {
                        chip_ast: chip_data["chip_ast"].clone(),
                        return_dt: chip_data["return_dt"].clone(),
                    });
                }
                // Register externals
                for (name, ext_data) in externals {
                    gen.registered_externals.insert(name.clone(), RegisteredExternal {
                        return_dt: ext_data["return_dt"].clone(),
                    });
                }
                Ok(gen.generate(&circuit))
            }
            _ => Err("Expected ASTCircuit at top level".to_string()),
        }
    }

    // ── Visitor dispatch ──────────────────────────────────────────────

    fn visit(&mut self, node: &ASTNode) -> Value {
        match node {
            ASTNode::ASTCircuit(n) => { self.visit_circuit(n); Value::None }
            ASTNode::ASTAssignStatement(n) => { self.visit_assign(n); Value::None }
            ASTNode::ASTAugAssignStatement(n) => { self.visit_aug_assign(n); Value::None }
            ASTNode::ASTCondStatement(n) => { self.visit_cond(n); Value::None }
            ASTNode::ASTForInStatement(n) => { self.visit_for_in(n); Value::None }
            ASTNode::ASTWhileStatement(n) => { self.visit_while(n); Value::None }
            ASTNode::ASTBreakStatement(_) => { self.visit_break(); Value::None }
            ASTNode::ASTContinueStatement(_) => { self.visit_continue(); Value::None }
            ASTNode::ASTReturnStatement(n) => { self.visit_return(n); Value::None }
            ASTNode::ASTAssertStatement(n) => { self.visit_assert(n); Value::None }
            ASTNode::ASTPassStatement(_) => Value::None,
            ASTNode::ASTExprStatement(n) => self.visit(&n.expr),
            ASTNode::ASTBinaryOperator(n) => self.visit_binary_op(n),
            ASTNode::ASTUnaryOperator(n) => self.visit_unary_op(n),
            ASTNode::ASTNamedAttribute(n) => self.visit_named_attr(n),
            ASTNode::ASTExprAttribute(n) => self.visit_expr_attr(n),
            ASTNode::ASTLoad(n) => self.visit_load(n),
            ASTNode::ASTSubscriptExp(n) => self.visit_subscript(n),
            ASTNode::ASTConstantFloat(n) => self.builder.ir_constant_float(n.value),
            ASTNode::ASTConstantInteger(n) => self.builder.ir_constant_int(n.value),
            ASTNode::ASTConstantBoolean(n) => self.builder.ir_constant_bool(n.value),
            ASTNode::ASTConstantComplex(n) => {
                let real = self.builder.ir_constant_float(n.real);
                let imag = self.builder.ir_constant_float(n.imag);
                let real_sv = match real {
                    crate::types::Value::Float(s) => s,
                    _ => unreachable!("ir_constant_float returns Value::Float"),
                };
                let imag_sv = match imag {
                    crate::types::Value::Float(s) => s,
                    _ => unreachable!("ir_constant_float returns Value::Float"),
                };
                crate::types::Value::Complex { real: real_sv, imag: imag_sv }
            }
            ASTNode::ASTConstantNone(_) => Value::None,
            ASTNode::ASTConstantString(n) => self.builder.ir_constant_str(n.value.clone()),
            ASTNode::ASTSquareBrackets(n) => self.visit_square_brackets(n),
            ASTNode::ASTParenthesis(n) => self.visit_parenthesis(n),
            ASTNode::ASTGeneratorExp(n) => self.visit_generator_exp(n),
            ASTNode::ASTCondExp(n) => self.visit_cond_exp(n),
            ASTNode::ASTJoinedStr(n) => self.visit_joined_str(n),
            ASTNode::ASTFormattedValue(n) => self.visit_formatted_value(n),
            ASTNode::ASTStarredExpr(_) => panic!("Can't use starred expression here"),
            // Assignment targets are not visited as expressions
            ASTNode::ASTNameAssignTarget(_)
            | ASTNode::ASTSubscriptAssignTarget(_)
            | ASTNode::ASTTupleAssignTarget(_)
            | ASTNode::ASTListAssignTarget(_)
            | ASTNode::ASTChip(_) => {
                panic!("Cannot visit {:?} as expression", std::mem::discriminant(node))
            }
        }
    }

    // ── Circuit ───────────────────────────────────────────────────────

    fn visit_circuit(&mut self, n: &ASTCircuit) {
        // Process inputs
        for (_i, inp) in n.inputs.iter().enumerate() {
            let dt = self.parse_dt_descriptor(&inp.annotation.dt);
            let kind = inp.annotation.kind.as_deref().unwrap_or("Private");
            let is_public = kind == "Public";
            let val = self.read_input_value(&dt, &inp.name, vec![], is_public);
            self.ctx.set(&inp.name, val);
        }

        self.register_global_datatypes();

        for stmt in &n.block {
            self.visit(stmt);
        }
    }

    // ── Statements ────────────────────────────────────────────────────
}
