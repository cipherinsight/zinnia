//! The AST walker: the statement judgment of `definition.md` §4.
//!
//! ```text
//! ⟨Env, φ, Φ⟩ ⊢ s ⇒ ⟨Env', φ', Φ'⟩
//! ```
//!
//! The walker mirrors `IRGenerator::visit` (`src/ir_gen/mod.rs`) variant for
//! variant: one handler per `ASTNode`, each producing a descriptor, changing
//! the state, or returning a [`Reject`]. It knows no operator semantics —
//! every shape comes from a rule of [`crate::phase0::rules`] and every name
//! from the table in [`crate::phase0::ops`].
//!
//! | Module | Content |
//! |---|---|
//! | [`env`] | the name environment and the `if` merge of §6.1 |
//! | [`stmts`] | `S-Assign`, `S-If`, `S-Return`, `S-Assert`, `S-Unpack`, `S-SetItem` (§5.10, §5.11, §6.1) |
//! | [`constructors`] | the constructors of §5.8 and the value-sized lengths of §5.9 |
//! | [`loops`] | `S-For-Unrolled`, `S-For-Elems`, `S-For-Dyn`, `S-While-Bounded` (§6.2) |
//! | [`while_cap`] | the structural cap `S-While-Bounded` takes from the enclosing `for` (§6.2) |
//! | [`ranges`] | the `range` iterable of §6.2, constant and bounded |
//! | [`exprs`] | the expression judgment: operators, composites, `E-CondExp`, `E-Comp` |
//! | [`subscript`] | the subscript forms of §5.4 and §5.5, for reads and for `S-SetItem` |
//! | [`calls`] | `S-Call` (chip inlining), `E-External`, and the operator dispatch |
//!
//! # Evaluation order
//!
//! Left to right, arguments before the call, as Python evaluates and as the
//! baseline evaluates where it is a faithful Python evaluator. Where the
//! baseline deviates (`coverage.md` §9) the walker follows Python.
//!
//! # The path condition
//!
//! `φ` is the conjunction of three things, as in the baseline's
//! `IRContext::get_condition_value`: the branch conditions in force
//! ([`Walker::branch`]), the liveness of every enclosing loop (a `break` or
//! `continue` removes the conditions it fired under), and the negation of
//! every `return` already taken in this frame. A statement whose `φ` folds to
//! `false` is dead and is not walked.

pub(crate) mod arrays;
pub(crate) mod calls;
pub(crate) mod constructors;
pub(crate) mod env;
pub(crate) mod exprs;
pub(crate) mod gather;
pub(crate) mod loops;
pub(crate) mod ranges;
pub(crate) mod stmts;
pub(crate) mod subscript;
pub(crate) mod values;
pub(crate) mod while_cap;

use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::{ASTChip, ASTCircuit, ASTNode};
use crate::phase0::desc::{Desc, ValueKey};
use crate::phase0::rules::inputs::ContractSymbols;
use crate::phase0::rules::RuleCtx;
use crate::phase0::source::{Producer, ValueFrame};
use crate::phase0::term::BoolId;
use crate::phase0::{Analysis, Phase0Config, Phase0Result, Reject};

use env::Frame;
use while_cap::WhileCap;

/// The expression index the delivery set's descriptor is recorded under.
///
/// Circuit inputs occupy the low indices of the empty statement path
/// (`analyze` keys input `k` as `at(k)`), so the returned value takes the
/// highest index instead of colliding with one of them.
const DELIVERY_EXPR_INDEX: u32 = u32::MAX;

/// The names the baseline binds in every scope before the body runs
/// (`IRGenerator::register_global_datatypes`), as dtype objects.
///
/// The Boolean names are bound to `ZinniaType::Boolean`, where the baseline
/// binds them to `Integer` (`coverage.md` §4 records the alias). The alias
/// loses the one thing a shape pass needs from `np.ones(n, dtype=bool)`: that
/// the result is a *mask*, so `a[m]` is `E-Filter` with a count and not
/// `E-Take` with the index array's length. Phase 0 follows NumPy here, as it
/// does for `np.in1d` (`coverage.md` §9).
const GLOBAL_DTYPES: [(&str, crate::types::ZinniaType); 13] = [
    ("Float", crate::types::ZinniaType::Float),
    ("float", crate::types::ZinniaType::Float),
    ("Integer", crate::types::ZinniaType::Integer),
    ("int", crate::types::ZinniaType::Integer),
    ("Int", crate::types::ZinniaType::Integer),
    ("integer", crate::types::ZinniaType::Integer),
    ("Boolean", crate::types::ZinniaType::Boolean),
    ("bool", crate::types::ZinniaType::Boolean),
    ("Bool", crate::types::ZinniaType::Boolean),
    ("boolean", crate::types::ZinniaType::Boolean),
    ("Complex", crate::types::ZinniaType::Complex),
    ("complex", crate::types::ZinniaType::Complex),
    ("np", crate::types::ZinniaType::Class),
];

/// Whether control left the current block (`IRGenerator`'s
/// `check_return_guaranteed` / `check_loop_terminated_guaranteed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flow {
    Normal,
    /// A `return`, `break` or `continue` that definitely ran: the rest of the
    /// block is unreachable.
    Terminated,
}

/// The liveness of one enclosing loop (§6.2).
#[derive(Debug, Clone, Copy)]
pub(crate) struct LoopGuard {
    /// Removed by `break`; persists across iterations.
    live: BoolId,
    /// Removed by `continue`; reset at the start of every iteration.
    iter_live: BoolId,
}

/// A chip, parsed once and shared by every call site (§5.10).
pub(crate) struct ChipEntry {
    pub chip: Rc<ASTChip>,
    pub return_dt: Option<crate::types::ZinniaType>,
}

/// The forward pass.
pub(crate) struct Walker<'a> {
    pub an: Analysis,
    pub symbols: ContractSymbols,
    pub config: Phase0Config,
    chips: &'a HashMap<String, serde_json::Value>,
    externals: &'a HashMap<String, serde_json::Value>,
    parsed_chips: HashMap<String, ChipEntry>,
    frames: Vec<Frame>,
    /// The branch conditions currently in force (§2's `φ`, branch part).
    branch: BoolId,
    loops: Vec<LoopGuard>,
    /// The structural cap of `S-While-Bounded` (§6.2), which is pure syntax
    /// over the enclosing `for` loops.
    pub(crate) while_cap: WhileCap,
    /// The number of chip bodies currently inlined (§5.10's recursion depth).
    depth: u32,
    /// Work done so far, against [`Phase0Config::work_budget`].
    visited: u64,
}

impl<'a> Walker<'a> {
    pub fn new(
        an: Analysis,
        symbols: ContractSymbols,
        config: Phase0Config,
        chips: &'a HashMap<String, serde_json::Value>,
        externals: &'a HashMap<String, serde_json::Value>,
        inputs: Vec<(String, Desc)>,
    ) -> Self {
        let branch = an.terms.tt();
        let mut circuit_frame = Frame::new();
        for (position, (name, desc)) in inputs.into_iter().enumerate() {
            // Input `k` is keyed as `at(k)` before the walk starts
            // (`crate::phase0::build_input_descriptors`), and that key is the
            // binding's, so a use of the name aliases it.
            circuit_frame.set(&name, desc, Some(an.keys.at(position as u32)));
        }
        let mut walker = Walker {
            an,
            symbols,
            config,
            chips,
            externals,
            parsed_chips: HashMap::new(),
            frames: vec![circuit_frame],
            branch,
            loops: Vec::new(),
            while_cap: WhileCap::default(),
            depth: 0,
            visited: 0,
        };
        walker.register_global_datatypes();
        walker
    }

    /// Bind the dtype names the baseline registers in every scope, so that
    /// `a.astype(float)` and `np.array(v, dtype=int)` see a value.
    fn register_global_datatypes(&mut self) {
        for (name, dt) in GLOBAL_DTYPES {
            if self.frame().get(name).is_none() {
                // Not a value of Δ: a dtype name has no key, and a use of one
                // is an opaque value (`implementation.md` §5a).
                self.frame_mut().set(name, Desc::Class(dt), None);
            }
        }
    }

    /// Walk the circuit body and finish the pass: the returned values merge
    /// by `S-If` (§5.10) into the delivery set `R` (§1).
    pub fn run(mut self, circuit: &ASTCircuit) -> Result<Phase0Result, Reject> {
        self.block(&circuit.block)?;
        // The frame stays on the stack: merging the returns evaluates rules,
        // which need a path condition, which reads the frame.
        let returns = std::mem::take(&mut self.frame_mut().returns);
        // The delivery exists exactly when the circuit returned. `φ` at the
        // end of the frame excludes every return already taken, so it is the
        // wrong condition to give it (§2), and the merged descriptor carries
        // the ψ of whichever return built it.
        let delivered_when = match self.frame().returned {
            Some(condition) => condition,
            None => self.an.terms.tt(),
        };
        let key = ValueKey {
            stmt_path: Vec::new(),
            iters: Vec::new(),
            calls: Vec::new(),
            expr_index: DELIVERY_EXPR_INDEX,
        };
        let (merged, frame) =
            self.in_value(&key, |w| stmts::merge_returns(w, returns, &key, "return"))?;
        if !matches!(merged.desc, Desc::None) {
            let producer = merged.producer(frame.into_rules());
            let desc = created_at(merged.desc, delivered_when);
            self.record(key.clone(), Some("return"), &desc, producer);
            self.an.delivery.push(key);
        }
        Ok(self.an.finish())
    }

    // ── The abstract state ────────────────────────────────────────────

    pub fn frame(&self) -> &Frame {
        self.frames
            .last()
            .expect("a frame is pushed at construction")
    }

    pub fn frame_mut(&mut self) -> &mut Frame {
        self.frames
            .last_mut()
            .expect("a frame is pushed at construction")
    }

    /// The liveness part of `φ`: no enclosing loop has been left and no
    /// `return` of this frame has fired.
    pub fn live(&mut self) -> BoolId {
        let guards: Vec<BoolId> = self
            .loops
            .iter()
            .flat_map(|g| [g.live, g.iter_live])
            .collect();
        let mut live = self.an.terms.all(&guards);
        if let Some(returned) = self.frame().returned {
            let not_returned = self.an.terms.not(returned);
            live = self.an.terms.and(live, not_returned);
        }
        live
    }

    /// `φ`, the path condition of §2.
    pub fn path(&mut self) -> BoolId {
        let live = self.live();
        let branch = self.branch;
        self.an.terms.and(branch, live)
    }

    /// The rule context at the current path condition.
    pub fn ctx(&mut self) -> RuleCtx<'_> {
        let path = self.path();
        self.an.ctx(path)
    }

    /// Whether the current path condition is syntactically `false`, i.e. the
    /// statement is unreachable.
    pub fn is_dead(&mut self) -> bool {
        let path = self.path();
        self.an.terms.is_false(path)
    }

    // ── Blocks and statements ─────────────────────────────────────────

    /// Walk a block of statements, stopping at the first that terminates
    /// control or once the path condition has folded to `false`.
    pub fn block(&mut self, stmts: &[ASTNode]) -> Result<Flow, Reject> {
        let mark = self.an.keys.enter_block();
        let mut flow = Flow::Normal;
        for (index, stmt) in stmts.iter().enumerate() {
            self.spend(1)?;
            self.an.keys.set_stmt(index as u32);
            flow = self.stmt(stmt)?;
            if flow == Flow::Terminated || self.is_dead() {
                flow = Flow::Terminated;
                break;
            }
        }
        self.an.keys.exit_block(mark);
        Ok(flow)
    }

    /// Charge `units` of work against [`Phase0Config::work_budget`].
    ///
    /// One unit is one statement, one expression node, one unrolled
    /// iteration, or one descriptor leaf merged at a dynamic `if`. The budget
    /// turns a program that would not finish — a loop nest whose trip counts
    /// are declared maxima rather than compilation parameters (§6.2) — into a
    /// rejection the coverage report can name.
    pub fn spend(&mut self, units: u64) -> Result<(), Reject> {
        self.visited = self.visited.saturating_add(units);
        if self.visited > self.config.work_budget {
            return Err(Reject::BudgetExceeded {
                units: self.visited,
                budget: self.config.work_budget,
            });
        }
        Ok(())
    }

    /// One statement. Mirrors the statement arms of `IRGenerator::visit`.
    fn stmt(&mut self, node: &ASTNode) -> Result<Flow, Reject> {
        match node {
            ASTNode::ASTAssignStatement(n) => stmts::assign(self, n),
            ASTNode::ASTCondStatement(n) => stmts::cond(self, n),
            ASTNode::ASTForInStatement(n) => loops::for_in(self, n),
            ASTNode::ASTWhileStatement(n) => loops::while_loop(self, n),
            ASTNode::ASTBreakStatement(_) => self.loop_exit(true),
            ASTNode::ASTContinueStatement(_) => self.loop_exit(false),
            ASTNode::ASTReturnStatement(n) => stmts::return_stmt(self, n),
            ASTNode::ASTAssertStatement(n) => stmts::assert_stmt(self, n),
            ASTNode::ASTPassStatement(_) => Ok(Flow::Normal),
            ASTNode::ASTExprStatement(n) => {
                self.expr(&n.expr)?;
                Ok(Flow::Normal)
            }
            // The front end desugars `x op= v` into an assignment
            // (`coverage.md` §2), so this node never reaches phase 0; the
            // baseline drops the write (`coverage.md` §9).
            ASTNode::ASTAugAssignStatement(_) => Err(Reject::Unsupported {
                construct: "augmented assignment (the front end desugars it)".to_string(),
            }),
            other => Err(Reject::Unsupported {
                construct: format!("{} in statement position", node_name(other)),
            }),
        }
    }

    // ── Control flow ──────────────────────────────────────────────────

    /// `break` (`exit_loop`) and `continue`: the condition they fire under
    /// leaves the loop's liveness (§6.2).
    fn loop_exit(&mut self, is_break: bool) -> Result<Flow, Reject> {
        let path = self.path();
        let not_path = self.an.terms.not(path);
        let guard = self.loops.last_mut().ok_or_else(|| Reject::Unsupported {
            construct: format!(
                "`{}` outside a loop",
                if is_break { "break" } else { "continue" }
            ),
        })?;
        if is_break {
            guard.live = and_raw(&mut self.an, guard.live, not_path);
        } else {
            guard.iter_live = and_raw(&mut self.an, guard.iter_live, not_path);
        }
        Ok(Flow::Terminated)
    }

    /// Run `body` with `condition` added to the branch part of `φ`, then
    /// restore it (§6.1).
    pub fn under_branch<T>(
        &mut self,
        condition: BoolId,
        body: impl FnOnce(&mut Self) -> Result<T, Reject>,
    ) -> Result<T, Reject> {
        let saved = self.push_branch(condition);
        let out = body(self);
        self.restore_branch(saved);
        out
    }

    /// Add `condition` to the branch part of `φ`, returning the value to
    /// restore.
    fn push_branch(&mut self, condition: BoolId) -> BoolId {
        let saved = self.branch;
        self.branch = self.an.terms.and(saved, condition);
        saved
    }

    fn restore_branch(&mut self, saved: BoolId) {
        self.branch = saved;
    }

    /// Open a loop: its liveness is `true` until a `break` fires (§6.2).
    pub fn enter_loop(&mut self) {
        let tt = self.an.terms.tt();
        self.loops.push(LoopGuard {
            live: tt,
            iter_live: tt,
        });
    }

    /// Start one iteration: `continue` only affects the iteration it fired
    /// in.
    pub fn start_iteration(&mut self) {
        let tt = self.an.terms.tt();
        if let Some(guard) = self.loops.last_mut() {
            guard.iter_live = tt;
        }
    }

    /// Run the rest of this iteration only when `condition` holds: the guard
    /// of `S-For-Dyn`, `S-For-Range-Bounded` and `S-While-Bounded` (§6.2).
    ///
    /// The guard joins the *liveness* part of `φ`, not the branch part.
    /// Both would put it in `φ` and so guard every check, but only liveness
    /// also reaches the bindings: [`Walker::bind_name`] merges a new binding
    /// against the previous one under the liveness condition, which is what
    /// makes `s` after an iteration that may not have run `ite(guard, new,
    /// old)`. The branch part is deliberately left out there, because §6.1's
    /// merge at the end of an `if` handles it — and a guarded iteration has
    /// no such merge.
    pub fn guard_iteration(&mut self, condition: BoolId) {
        let tightened = match self.loops.last() {
            Some(guard) => self.an.terms.and(guard.iter_live, condition),
            None => return,
        };
        if let Some(guard) = self.loops.last_mut() {
            guard.iter_live = tightened;
        }
    }

    /// Whether the loop has been left by a `break` on every path, so no
    /// further iteration can run.
    pub fn loop_is_dead(&mut self) -> bool {
        match self.loops.last() {
            Some(guard) => {
                let live = guard.live;
                self.an.terms.is_false(live) || self.is_dead()
            }
            None => false,
        }
    }

    /// Close a loop and return the condition under which it ran to
    /// completion, which is what Python's `for ... else` / `while ... else`
    /// runs under.
    pub fn exit_loop(&mut self) -> BoolId {
        match self.loops.pop() {
            Some(guard) => guard.live,
            None => self.an.terms.tt(),
        }
    }

    // ── Frames ────────────────────────────────────────────────────────

    /// Enter an inlined chip body with a fresh environment (§5.10).
    pub fn push_frame(&mut self) -> Result<(), Reject> {
        if self.depth >= self.config.recursion_limit {
            return Err(Reject::RecursionTooDeep {
                limit: self.config.recursion_limit,
            });
        }
        self.depth += 1;
        self.frames.push(Frame::new());
        self.register_global_datatypes();
        Ok(())
    }

    pub fn pop_frame(&mut self) -> Frame {
        self.depth = self.depth.saturating_sub(1);
        self.frames
            .pop()
            .expect("push_frame and pop_frame are paired")
    }

    pub fn chip(&mut self, name: &str) -> Option<Result<&ChipEntry, Reject>> {
        if !self.parsed_chips.contains_key(name) {
            let raw = self.chips.get(name)?;
            match parse_chip(raw) {
                Ok(entry) => {
                    self.parsed_chips.insert(name.to_string(), entry);
                }
                Err(reject) => return Some(Err(reject)),
            }
        }
        self.parsed_chips.get(name).map(Ok)
    }

    pub fn is_chip(&self, name: &str) -> bool {
        self.chips.contains_key(name)
    }

    /// The declared return type of an `@zk_external` function (§5.11).
    pub fn external_return_dt(
        &self,
        name: &str,
    ) -> Option<Result<crate::types::ZinniaType, Reject>> {
        let raw = self.externals.get(name)?;
        Some(
            serde_json::from_value(raw["return_dt"].clone()).map_err(|e| Reject::Unsupported {
                construct: format!("external `{}` has an unreadable return type: {}", name, e),
            }),
        )
    }

    // ── Δ and R ───────────────────────────────────────────────────────

    /// Record one value in `Δ` under `key`, with the producer that made it.
    ///
    /// Every keyed value gets an entry (§1: "Δ maps every SSA value"); the
    /// name is `Some` only where the value has one — a circuit input, a bound
    /// name, the delivery set. See "Value keys, as implemented" in
    /// `implementation.md`.
    pub fn record(&mut self, key: ValueKey, name: Option<&str>, desc: &Desc, producer: Producer) {
        self.an
            .record(key, name.map(str::to_string), desc.clone(), producer);
    }

    /// Evaluate `body` as the value at `key`: the frame it opens collects the
    /// rules applied and the keys of the values they read, which become the
    /// value's producer, and `key` is the subject of every check recorded
    /// inside it (`implementation.md` §5a).
    ///
    /// The frame is closed whether or not the body rejects, so the stack
    /// stays balanced on the way out of a rejection.
    pub fn in_value<T>(
        &mut self,
        key: &ValueKey,
        body: impl FnOnce(&mut Self) -> Result<T, Reject>,
    ) -> Result<(T, ValueFrame), Reject> {
        self.an.open_value(key);
        let out = body(self);
        let frame = self.an.close_value();
        Ok((out?, frame))
    }

    /// Run `body` with `key` as the subject of the checks it records, without
    /// producing a value: the statement-level rule applications of §5a
    /// (`S-Assert`, the `while` cap).
    pub fn with_subject<T>(
        &mut self,
        key: &ValueKey,
        body: impl FnOnce(&mut Self) -> Result<T, Reject>,
    ) -> Result<T, Reject> {
        let (out, _) = self.in_value(key, body)?;
        Ok(out)
    }

    /// Run `body` with the iterated value as the subject when it has a key,
    /// and in the enclosing value's frame otherwise.
    ///
    /// An iterated value has no key of its own where it is not an expression
    /// of the program: the elements a starred argument spreads, for instance,
    /// belong to the value that was spread (`implementation.md` §7).
    pub fn in_iterable<T>(
        &mut self,
        key: Option<&ValueKey>,
        body: impl FnOnce(&mut Self) -> Result<T, Reject>,
    ) -> Result<T, Reject> {
        match key {
            Some(key) => self.with_subject(key, body),
            None => body(self),
        }
    }
}

/// The entries of an argument that is one value or a sequence of them.
///
/// A shape, an axis list, a pad-width list and a reshape target are all
/// spelled either way in NumPy — `a.reshape(6)` and `a.reshape((2, 3))`,
/// `np.sum(a, axis=0)` and `np.sum(a, axis=(0, 1))` — and every rule that
/// reads one starts here and then converts each entry to what it needs.
pub(crate) fn entries(value: &Desc) -> Vec<Desc> {
    match value.as_composite() {
        Some(c) => c.elems.clone(),
        None => vec![value.clone()],
    }
}

/// The same descriptor with $\psi$ set to `path`: the execution condition of
/// a value created in the current state (§2).
///
/// Every entry of $\Delta$ carries it, because §5a rule 2 keys an expression
/// *occurrence*: a use of a name inside a branch is executed only under that
/// branch, whatever the $\psi$ of the value it names. A composite has no
/// $\psi$ of its own (§3.3), so the condition reaches its elements instead,
/// and a value with no descriptor content has none to set.
pub(super) fn created_at(desc: Desc, path: BoolId) -> Desc {
    match desc {
        Desc::Scalar(s) => Desc::Scalar(crate::phase0::desc::ScalarDesc { psi: path, ..s }),
        Desc::Array(a) => Desc::Array(crate::phase0::desc::ArrayDesc { psi: path, ..a }),
        Desc::Composite(c) => Desc::Composite(crate::phase0::desc::CompositeDesc {
            kind: c.kind,
            elems: c.elems.into_iter().map(|e| created_at(e, path)).collect(),
        }),
        other => other,
    }
}

/// `a ∧ b` on the arena behind an `Analysis`, for callers that hold a field
/// borrow and cannot go through [`Walker::ctx`].
fn and_raw(an: &mut Analysis, a: BoolId, b: BoolId) -> BoolId {
    an.terms.and(a, b)
}

fn parse_chip(raw: &serde_json::Value) -> Result<ChipEntry, Reject> {
    let node: ASTNode =
        serde_json::from_value(raw["chip_ast"].clone()).map_err(|e| Reject::Unsupported {
            construct: format!("unreadable chip AST: {}", e),
        })?;
    let chip = match node {
        ASTNode::ASTChip(chip) => Rc::new(chip),
        _ => {
            return Err(Reject::Unsupported {
                construct: "a chip whose AST is not an ASTChip".to_string(),
            })
        }
    };
    let return_dt = if raw["return_dt"].is_null() {
        None
    } else {
        Some(
            serde_json::from_value(raw["return_dt"].clone()).map_err(|e| Reject::Unsupported {
                construct: format!("unreadable chip return type: {}", e),
            })?,
        )
    };
    Ok(ChipEntry { chip, return_dt })
}

/// The node's class name, for rejection messages, and nothing else: it
/// re-spells the `#[serde(tag = "__class__")]` names of `src/ast/mod.rs` so
/// that a construct phase 0 has no rule for can be named. The match is
/// exhaustive, so it cannot drift silently. It sits at the bottom of the
/// module because it is noise between the walker's own items.
pub(crate) fn node_name(node: &ASTNode) -> &'static str {
    match node {
        ASTNode::ASTCircuit(_) => "ASTCircuit",
        ASTNode::ASTChip(_) => "ASTChip",
        ASTNode::ASTAssignStatement(_) => "ASTAssignStatement",
        ASTNode::ASTAugAssignStatement(_) => "ASTAugAssignStatement",
        ASTNode::ASTCondStatement(_) => "ASTCondStatement",
        ASTNode::ASTForInStatement(_) => "ASTForInStatement",
        ASTNode::ASTWhileStatement(_) => "ASTWhileStatement",
        ASTNode::ASTBreakStatement(_) => "ASTBreakStatement",
        ASTNode::ASTContinueStatement(_) => "ASTContinueStatement",
        ASTNode::ASTReturnStatement(_) => "ASTReturnStatement",
        ASTNode::ASTAssertStatement(_) => "ASTAssertStatement",
        ASTNode::ASTPassStatement(_) => "ASTPassStatement",
        ASTNode::ASTExprStatement(_) => "ASTExprStatement",
        ASTNode::ASTBinaryOperator(_) => "ASTBinaryOperator",
        ASTNode::ASTUnaryOperator(_) => "ASTUnaryOperator",
        ASTNode::ASTNamedAttribute(_) => "ASTNamedAttribute",
        ASTNode::ASTExprAttribute(_) => "ASTExprAttribute",
        ASTNode::ASTLoad(_) => "ASTLoad",
        ASTNode::ASTSubscriptExp(_) => "ASTSubscriptExp",
        ASTNode::ASTConstantFloat(_) => "ASTConstantFloat",
        ASTNode::ASTConstantInteger(_) => "ASTConstantInteger",
        ASTNode::ASTConstantBoolean(_) => "ASTConstantBoolean",
        ASTNode::ASTConstantComplex(_) => "ASTConstantComplex",
        ASTNode::ASTConstantNone(_) => "ASTConstantNone",
        ASTNode::ASTConstantString(_) => "ASTConstantString",
        ASTNode::ASTSquareBrackets(_) => "ASTSquareBrackets",
        ASTNode::ASTParenthesis(_) => "ASTParenthesis",
        ASTNode::ASTGeneratorExp(_) => "ASTGeneratorExp",
        ASTNode::ASTCondExp(_) => "ASTCondExp",
        ASTNode::ASTJoinedStr(_) => "ASTJoinedStr",
        ASTNode::ASTFormattedValue(_) => "ASTFormattedValue",
        ASTNode::ASTStarredExpr(_) => "ASTStarredExpr",
        ASTNode::ASTNameAssignTarget(_) => "ASTNameAssignTarget",
        ASTNode::ASTSubscriptAssignTarget(_) => "ASTSubscriptAssignTarget",
        ASTNode::ASTTupleAssignTarget(_) => "ASTTupleAssignTarget",
        ASTNode::ASTListAssignTarget(_) => "ASTListAssignTarget",
    }
}
