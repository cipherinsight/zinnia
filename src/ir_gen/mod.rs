//! IR Generator — visitor pattern over Zinnia AST, produces IRGraph.
//! Ports `zinnia/compile/ir/ir_gen.py` (854 lines).

use std::collections::HashMap;

use crate::ast::*;
use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_ctx::IRContext;
use crate::optim::predicates::ContractTerm;
use crate::optim::prove::{maximize_shape_param, minimize_shape_param};
use crate::optim::resolver::{LayeredResolver, Resolver, StaticOnlyResolver};
use crate::types::{NumberType, ShapeDim, Value, ZinniaType};

mod chip_external;
mod detached_copies;
mod helpers;
pub mod input_domain;
mod iteration;
mod list_methods;
mod list_sharing;
mod named_attr;
pub mod outputs;
mod visitors;

#[cfg(test)]
mod tests;

/// Whether `node` is a statement (as opposed to an expression or target).
fn is_statement(node: &ASTNode) -> bool {
    matches!(
        node,
        ASTNode::ASTAssignStatement(_)
            | ASTNode::ASTAugAssignStatement(_)
            | ASTNode::ASTCondStatement(_)
            | ASTNode::ASTForInStatement(_)
            | ASTNode::ASTWhileStatement(_)
            | ASTNode::ASTBreakStatement(_)
            | ASTNode::ASTContinueStatement(_)
            | ASTNode::ASTReturnStatement(_)
            | ASTNode::ASTAssertStatement(_)
            | ASTNode::ASTPassStatement(_)
            | ASTNode::ASTExprStatement(_)
    )
}

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
    pub chip_ast: serde_json::Value,          // The ASTChip dict
    pub return_dt: Option<serde_json::Value>, // Return type descriptor
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
    pub(crate) for_loop_frames: Vec<ForLoopFrame>,
    pub(crate) induction_origins: HashMap<String, String>,
    pub(crate) loop_index_bound_contexts: Vec<LoopIndexBoundContext>,
    pub(crate) loop_read_reuse_contexts: Vec<LoopReadReuseContext>,
    /// Copies made where NumPy would return a view (see `detached_copies`).
    pub(crate) detached_copies: detached_copies::DetachedCopies,
    /// Where every public output value comes from (see `outputs`).
    pub(crate) output_layout: outputs::OutputLayout,
    /// The array arguments of an in-place kernel (one that returns no
    /// value), as bound at entry; empty otherwise.
    pub(crate) in_place_arguments: Vec<outputs::ArgumentArray>,
    /// Index of the first IR statement of the entry function's body.
    pub(crate) in_place_body_start: usize,
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

/// IR-gen-local structural loop context. This is deliberately not a resolver
/// fact or exported IR semantic; it only tightens while unroll quotas for
/// exact structural patterns.
#[derive(Debug, Clone)]
pub(crate) struct ForLoopFrame {
    pub var_name: String,
    pub max_bound: u32,
    pub current_bound: Option<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct LoopIndexBoundContext {
    pub var_name: String,
    pub max_bound: u32,
    pub current_bound: u32,
    pub positive_guard: bool,
    pub guard_condition: Option<Value>,
    pub record_guard_read: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LoopIndexExprKey {
    Var,
    MinusOne,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StaticArrayIdentity {
    pub dtype: NumberType,
    pub segment_id: u32,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct ReusableLoopRead {
    pub array: StaticArrayIdentity,
    pub index: LoopIndexExprKey,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct LoopReadReuseContext {
    pub candidate: Option<ReusableLoopRead>,
    pub body_prefix_active: bool,
}

impl Default for LoopReadReuseContext {
    fn default() -> Self {
        Self {
            candidate: None,
            body_prefix_active: true,
        }
    }
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
            for_loop_frames: Vec::new(),
            induction_origins: HashMap::new(),
            loop_index_bound_contexts: Vec::new(),
            loop_read_reuse_contexts: Vec::new(),
            detached_copies: detached_copies::DetachedCopies::default(),
            output_layout: outputs::OutputLayout::default(),
            in_place_arguments: Vec::new(),
            in_place_body_start: 0,
        }
    }

    /// Generate an IRGraph from an AST circuit.
    pub fn generate(self, ast: &ASTCircuit) -> IRGraph {
        self.generate_program(ast).0
    }

    /// Main entry point: generate the IRGraph of an AST circuit and the
    /// layout of its outputs.
    pub fn generate_program(mut self, ast: &ASTCircuit) -> (IRGraph, outputs::OutputLayout) {
        self.visit_circuit(ast);
        let layout = std::mem::take(&mut self.output_layout);
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
        (graph, layout)
    }

    /// Generate from a JSON string (the bridge entry point).
    pub fn generate_from_json(config: IRGenConfig, ast_json: &str) -> Result<IRGraph, String> {
        let node: ASTNode =
            serde_json::from_str(ast_json).map_err(|e| format!("AST parse error: {}", e))?;
        match node {
            ASTNode::ASTCircuit(circuit) => {
                let gen = IRGenerator::new(config);
                Ok(gen.generate(&circuit))
            }
            _ => Err("Expected ASTCircuit at top level".to_string()),
        }
    }

    /// Generate from JSON with chips and externals; also returns the output
    /// layout.
    pub fn generate_from_json_with_chips(
        config: IRGenConfig,
        ast_json: &str,
        chips: &HashMap<String, serde_json::Value>,
        externals: &HashMap<String, serde_json::Value>,
    ) -> Result<(IRGraph, outputs::OutputLayout), String> {
        let node: ASTNode =
            serde_json::from_str(ast_json).map_err(|e| format!("AST parse error: {}", e))?;
        match node {
            ASTNode::ASTCircuit(circuit) => {
                let mut gen = IRGenerator::new(config);
                // Register chips
                for (name, chip_data) in chips {
                    gen.registered_chips.insert(
                        name.clone(),
                        RegisteredChip {
                            chip_ast: chip_data["chip_ast"].clone(),
                            return_dt: if chip_data["return_dt"].is_null() {
                                None
                            } else {
                                Some(chip_data["return_dt"].clone())
                            },
                        },
                    );
                }
                // Register externals
                for (name, ext_data) in externals {
                    gen.registered_externals.insert(
                        name.clone(),
                        RegisteredExternal {
                            return_dt: ext_data["return_dt"].clone(),
                        },
                    );
                }
                Ok(gen.generate_program(&circuit))
            }
            _ => Err("Expected ASTCircuit at top level".to_string()),
        }
    }

    // ── Visitor dispatch ──────────────────────────────────────────────

    /// Visit a node. A statement runs with the builder's write predicate set
    /// to its complete execution condition (see
    /// [`IRContext::get_condition_value`]): every memory write it makes,
    /// including writes made while evaluating its expressions or inside chips
    /// it calls, keeps the old cell on paths where the statement does not
    /// execute. Expressions inherit the predicate of their statement.
    fn visit(&mut self, node: &ASTNode) -> Value {
        if !is_statement(node) {
            return self.visit_inner(node);
        }
        let condition = self.ctx.get_condition_value(&mut self.builder);
        let previous = self.builder.swap_write_predicate(Some(condition));
        let result = self.visit_inner(node);
        self.builder.swap_write_predicate(previous);
        result
    }

    fn visit_inner(&mut self, node: &ASTNode) -> Value {
        match node {
            ASTNode::ASTCircuit(n) => {
                self.visit_circuit(n);
                Value::None
            }
            ASTNode::ASTAssignStatement(n) => {
                self.visit_assign(n);
                Value::None
            }
            // The Python transformer lowers every augmented assignment to a
            // plain assignment (`a op= b` becomes `a = a op b`).
            ASTNode::ASTAugAssignStatement(_) => {
                panic!("internal: augmented assignments must be lowered by the transformer")
            }
            ASTNode::ASTCondStatement(n) => {
                self.visit_cond(n);
                Value::None
            }
            ASTNode::ASTForInStatement(n) => {
                self.visit_for_in(n);
                Value::None
            }
            ASTNode::ASTWhileStatement(n) => {
                self.visit_while(n);
                Value::None
            }
            ASTNode::ASTBreakStatement(_) => {
                self.visit_break();
                Value::None
            }
            ASTNode::ASTContinueStatement(_) => {
                self.visit_continue();
                Value::None
            }
            ASTNode::ASTReturnStatement(n) => {
                self.visit_return(n);
                Value::None
            }
            ASTNode::ASTAssertStatement(n) => {
                self.visit_assert(n);
                Value::None
            }
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
                crate::types::Value::Complex {
                    real: real_sv,
                    imag: imag_sv,
                }
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
                panic!(
                    "Cannot visit {:?} as expression",
                    std::mem::discriminant(node)
                )
            }
        }
    }

    // ── Circuit ───────────────────────────────────────────────────────

    fn visit_circuit(&mut self, n: &ASTCircuit) {
        let scalar_preconditions = self.scalar_precondition_terms(n);
        let input_types = self.resolve_input_shape_types(n, &scalar_preconditions);

        let circuit_return_dt = n.return_dt.as_ref().map(|dt| self.parse_dt_descriptor(dt));

        self.ctx
            .circuit_enter(circuit_return_dt.clone().unwrap_or(ZinniaType::None));

        // Process inputs
        for (_i, inp) in n.inputs.iter().enumerate() {
            let dt = input_types
                .get(&inp.name)
                .cloned()
                .unwrap_or_else(|| self.parse_dt_descriptor(&inp.annotation.dt));
            let kind = inp.annotation.kind.as_deref().unwrap_or("Private");
            let is_public = kind == "Public";
            let val = self.read_input_value(&dt, &inp.name, vec![], is_public, inp.annotation.boolean);
            self.ctx.set(&inp.name, val);
        }

        self.register_global_datatypes();

        // Emit structural-predicate precondition atoms (compile-time facts
        // for the SMT layer) plus their witness-time enforcement (real
        // circuit constraints that the prover must satisfy).
        for req in &n.requires {
            self.builder.ir_structural_predicate(
                req.kind.clone(),
                req.args.clone(),
                req.op.clone(),
                req.bound.clone(),
            );
            // Witness-time enforcement: invoke the per-predicate emitter
            // if one is registered. Closes the soundness gap noted in
            // `compiler.nnz-witness-enforcement`.
            if let Some(emitter) =
                crate::optim::predicates::witness_emitters().get(req.kind.as_str())
            {
                // Resolve each predicate arg against the input bindings.
                // Args that are not context variables are numeric literals
                // baked into the predicate (e.g. the `0` / `5` bounds of
                // `forall(x, in_range(0, 5))`); materialise them as IR
                // constants. Previously these resolved to `Value::None` and
                // were silently coerced to constant 0 by `ensure_ptr`'s
                // (now-removed) lenient arm — which happened to match a `0`
                // bound but turned every non-zero literal bound (e.g. the
                // `hi = 5`) into a wrong `0`, a silent miscompile (C12b).
                let arg_values: Vec<crate::types::Value> = req
                    .args
                    .iter()
                    .map(|name| {
                        self.ctx.get(name).unwrap_or_else(|| {
                            match name.parse::<i64>() {
                                Ok(n) => self.builder.ir_constant_int(n),
                                Err(_) => crate::types::Value::None,
                            }
                        })
                    })
                    .collect();
                let bound_value = req.bound.as_ref().and_then(|name| {
                    self.ctx.get(name).or_else(|| {
                        // Numeric literal in the precondition (e.g.,
                        // `nnz(x) <= 1024`). Materialise as a constant.
                        name.parse::<i64>()
                            .ok()
                            .map(|n| self.builder.ir_constant_int(n))
                    })
                });
                (emitter.emit)(
                    &mut self.builder,
                    &arg_values,
                    req.op.as_deref(),
                    bound_value.as_ref(),
                );
            }
            // Op-contract enrichment: also deposit an `array_in_range`
            // fact on the input array's value_id so derived/mutated
            // arrays can pick up the bound without relying on the
            // payload-cache → input-name resolution path (which gets
            // invalidated by in-place writes / op rewrites).
            if req.kind == "forall_in_range" && req.args.len() == 3 {
                if let Some(arr_val) = self.ctx.get(&req.args[0]) {
                    if let (Some(arr_vid), Ok(lo), Ok(hi)) = (
                        arr_val.value_id(),
                        req.args[1].parse::<i64>(),
                        req.args[2].parse::<i64>(),
                    ) {
                        crate::helpers::static_array::reductions::deposit_array_in_range_fact(
                            &mut self.builder,
                            arr_vid,
                            lo,
                            hi,
                        );
                    }
                }
            }
        }

        // Emit scalar-precondition atoms + their witness-time enforcement.
        // The IR atom carries the serialized ContractTerm for the SMT
        // layer's discharger; the witness IR (computed via
        // `lower_precondition_to_ir`) is the prove-time enforcement.
        for scalar_req in &n.scalar_requires {
            // Deserialize ContractTerm from the JSON value and emit the
            // witness IR. A term that cannot be lowered is a compile error
            // (below): its fact is deposited first, and an unenforced fact
            // would let a malicious prover violate it.
            let term: Result<crate::optim::predicates::formula::ContractTerm, _> =
                serde_json::from_value(scalar_req.term.clone());
            let term = match term {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[scalar-precondition] malformed term: {e}");
                    continue;
                }
            };
            if !crate::optim::predicates::collect_shape_params(&term).is_empty() {
                continue;
            }
            // Serialize the term to a JSON string for the IR atom payload.
            let term_json = scalar_req.term.to_string();
            self.builder.ir_scalar_precondition(term_json);
            // Build a closure that resolves `Var(Input(name))` against
            // the circuit's input bindings. We snapshot input names
            // up front (from the AST) to avoid borrow-checker issues
            // when the closure later runs alongside `&mut self.builder`.
            let mut binds: std::collections::HashMap<String, crate::types::Value> =
                std::collections::HashMap::new();
            for inp in &n.inputs {
                if let Some(v) = self.ctx.get(&inp.name) {
                    binds.insert(inp.name.clone(), v);
                }
            }
            let lookup = |name: &str| binds.get(name).cloned();

            // Seed the scalar precondition into the FactStack
            // (compiler.eliminate-free-fact-bucket): substitute every
            // `Var(Input(name))` leaf with `Var(SsaPtr(p))` against the
            // parameter's SSA ptr (materialising one via `ensure_ptr` if
            // a parameter was a pure compile-time constant), then index
            // the resulting ptr-anchored term under every ptr it
            // references. Consumers (chokepoint fact-fallback,
            // `prove`) read `by_ptr[p]` and need not consult a separate
            // free bucket.
            let mut inputs_to_values: std::collections::HashMap<String, crate::types::ValueId> =
                std::collections::HashMap::new();
            for (name, val) in &binds {
                let with_ptr = self.builder.ensure_ptr(val);
                if let Some(vid) = with_ptr.value_id() {
                    inputs_to_values.insert(name.clone(), vid);
                }
            }
            let bound_term = crate::optim::predicates::substitute_inputs(&term, &inputs_to_values);
            let anchors = crate::optim::predicates::collect_value_ids(&bound_term);
            for anchor in &anchors {
                self.builder.facts.insert_for(*anchor, bound_term.clone());
            }

            match crate::optim::predicates::witness::lower_precondition_to_ir(
                &mut self.builder,
                &term,
                &lookup,
            ) {
                Ok(bool_val) => {
                    self.builder.ir_assert(&bool_val);
                }
                // The precondition is already a fact the optimizer relies
                // on; one the circuit cannot enforce must not compile.
                Err(e) => {
                    panic!("@requires precondition cannot be enforced in the circuit: {e}");
                }
            }
        }

        self.begin_in_place_kernel(n);
        for stmt in &n.block {
            self.visit(stmt);
        }

        let returns = self.ctx.get_returns_with_conditions();
        let return_guaranteed = self.ctx.check_return_guaranteed();
        if self.ctx.has_mixed_bare_and_value_returns() {
            panic!(
                "Circuit return inference does not allow mixing bare returns with value returns"
            );
        }
        let returns: Vec<(Value, Value)> = returns
            .into_iter()
            .filter(|(val, _)| !matches!(val, Value::None))
            .collect();
        if returns.is_empty() {
            if circuit_return_dt.as_ref().is_some_and(|dt| *dt != ZinniaType::None) {
                panic!("Circuit control ends without a value-returning statement");
            }
            // An in-place kernel: its outputs are the array arguments it wrote.
            for (source, value) in self.in_place_outputs() {
                self.expose_part(source, &value);
            }
            let _ = self.ctx.circuit_leave();
            return;
        }
        if !return_guaranteed {
            if circuit_return_dt.is_some() {
                panic!("Circuit return annotation requires every path to return");
            }
            panic!(
                "Circuit return inference does not support partial returns without an explicit return annotation"
            );
        }
        let mut result = returns[0].0.clone();
        for i in 1..returns.len() {
            let (val, cond) = &returns[i];
            result = crate::helpers::value_ops::select_value(&mut self.builder, cond, val, &result);
        }
        if let Some(return_dt) = circuit_return_dt {
            if result.zinnia_type() != return_dt {
                result = self.try_cast_value_to_match(result, &return_dt);
            }
            if result.zinnia_type() != return_dt {
                panic!(
                    "Circuit return type mismatch: expected {}, got {}",
                    return_dt,
                    result.zinnia_type()
                );
            }
        }
        self.expose_return_value(&result);
        let _ = self.ctx.circuit_leave();
    }

    fn scalar_precondition_terms(&self, n: &ASTCircuit) -> Vec<ContractTerm> {
        n.scalar_requires
            .iter()
            .filter_map(|scalar_req| serde_json::from_value(scalar_req.term.clone()).ok())
            .collect()
    }

    fn resolve_input_shape_types(
        &self,
        n: &ASTCircuit,
        facts: &[ContractTerm],
    ) -> HashMap<String, ZinniaType> {
        let mut out = HashMap::new();
        for inp in &n.inputs {
            let dt = self.parse_dt_descriptor(&inp.annotation.dt);
            let resolved = self.resolve_shape_type(&dt, facts);
            out.insert(inp.name.clone(), resolved);
        }
        out
    }

    fn resolve_shape_type(&self, dt: &ZinniaType, facts: &[ContractTerm]) -> ZinniaType {
        match dt {
            ZinniaType::SymbolicNDArray {
                shape,
                dtype,
                bindings,
            } => {
                let mut concrete_shape = Vec::with_capacity(shape.len());
                let mut max_shape = Vec::with_capacity(shape.len());
                let mut bounded_total = 1usize;
                let mut all_exact = true;
                for dim in shape {
                    match self.resolve_shape_dim(dim, bindings, facts) {
                        ResolvedShapeDim::Exact(n) => {
                            concrete_shape.push(n);
                            max_shape.push(n);
                            bounded_total = bounded_total.checked_mul(n).unwrap_or_else(|| {
                                panic!(
                                    "symbolic shape bound product overflows usize while resolving {:?}",
                                    shape
                                )
                            });
                        }
                        ResolvedShapeDim::Bounded(n) => {
                            all_exact = false;
                            max_shape.push(n);
                            bounded_total = bounded_total.checked_mul(n).unwrap_or_else(|| {
                                panic!(
                                    "symbolic shape bound product overflows usize while resolving {:?}",
                                    shape
                                )
                            });
                        }
                    }
                }
                if all_exact {
                    ZinniaType::NDArray {
                        shape: concrete_shape,
                        dtype: *dtype,
                    }
                } else {
                    ZinniaType::DynamicNDArray {
                        dtype: *dtype,
                        max_length: bounded_total,
                        max_rank: shape.len().max(1),
                        max_shape: Some(max_shape),
                    }
                }
            }
            _ => dt.clone(),
        }
    }

    fn resolve_shape_dim(
        &self,
        dim: &ShapeDim,
        bindings: &std::collections::BTreeMap<String, usize>,
        facts: &[ContractTerm],
    ) -> ResolvedShapeDim {
        match dim {
            ShapeDim::Const(n) => ResolvedShapeDim::Exact(*n),
            ShapeDim::Symbol(expr) => {
                if let Some(n) = bindings.get(expr) {
                    return ResolvedShapeDim::Exact(*n);
                }
                if let Some(n) = self.solve_shape_exact(expr, facts) {
                    return ResolvedShapeDim::Exact(n);
                }
                if let Some(n) = self.solve_shape_upper(expr, facts) {
                    return ResolvedShapeDim::Bounded(n);
                }
                panic!(
                    "symbolic shape parameter `{}` must be resolved to an exact value or finite bound before IR generation",
                    expr
                )
            }
            ShapeDim::Bounded { symbol, upper } => {
                if let Some(n) = bindings.get(symbol) {
                    ResolvedShapeDim::Exact(*n)
                } else {
                    ResolvedShapeDim::Bounded(*upper)
                }
            }
        }
    }

    fn solve_shape_exact(&self, symbol: &str, facts: &[ContractTerm]) -> Option<usize> {
        let min = minimize_shape_param(symbol, facts, 1, self.config.smt_query_timeout_ms)?;
        let max = maximize_shape_param(symbol, facts, 1, self.config.smt_query_timeout_ms)?;
        if min == max && min > 0 {
            Some(min as usize)
        } else {
            None
        }
    }

    fn solve_shape_upper(&self, symbol: &str, facts: &[ContractTerm]) -> Option<usize> {
        let max = maximize_shape_param(symbol, facts, 1, self.config.smt_query_timeout_ms)?;
        if max > 0 {
            Some(max as usize)
        } else {
            None
        }
    }

    // ── Statements ────────────────────────────────────────────────────
}

enum ResolvedShapeDim {
    Exact(usize),
    Bounded(usize),
}
