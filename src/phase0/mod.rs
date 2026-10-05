//! Phase 0: array-level abstract interpretation.
//!
//! Phase 0 is the first stage of the revised compiler
//! (`revision/phase0/definition.md` §1): a forward abstract interpretation of
//! the source program at the array level that maps a program and its
//! compilation parameters to $(\Phi, \Delta, R)$ or to a rejection. $\Phi$ is
//! a finite set of sorted variables with range constraints plus a finite set
//! of constraints over them; $\Delta$ maps every SSA value — array or scalar,
//! output or intermediate — to a descriptor recording its rank, dtype,
//! dimension terms, definedness condition and execution condition, together
//! with the rules that produced it and the values they read; $R$ is the
//! delivery set, the returned values the circuit must make observable. Every
//! check names the value its operation produced, so a consumer can tell which
//! operation instance a condition belongs to. It
//! computes a symbolic over-approximation of the program's shape behaviour —
//! lengths, counts, shapes, and success or failure — and models no element
//! values. It emits no wires, calls no solver, computes no numeric bound and
//! chooses no representation; the only numerals in its output come from the
//! compilation parameters.
//!
//! # Module layout
//!
//! | Module | Content |
//! |---|---|
//! | [`term`] | the hash-consed term language of §3.1, with evaluation, pretty, SMT-LIB and JSON printers |
//! | [`phi`] | variables, ranges, constraints, the `Count` memo of §5.5, product facts |
//! | [`desc`] | the descriptors of §3.2 and §3.3, and the value keys of `acceptance.md` §2.2 |
//! | [`source`] | the producer of every value: the source graph of §3.3 |
//! | [`interval`] | the syntactic upper bound of a term over $\Phi$'s ranges, for §5.9 and §6.2 |
//! | [`rules`] | one function per transfer rule of §5 and per merge rule of §6.1 |
//! | [`ops`] | the operator table: name to shape class (`coverage.md` §5) |
//! | `walk` | the traversal: the statement judgment of §4, the control flow of §6 |
//! | [`export`] | the JSON the Python harnesses of `acceptance.md` consume |
//!
//! # State of the implementation
//!
//! [`analyze`] parses the configuration, the contract facts and the input
//! annotations, builds the input descriptors of §5.1, and then walks the
//! circuit body, producing $(\Phi, \Delta, R)$ or a [`Reject`]. It is
//! complete for the parity set of `coverage.md` §1 except for the programs
//! that §9 records as depending on a baseline defect.
//!
//! The three acceptance harnesses of `acceptance.md` — coverage, adequacy
//! and consistency — are in `testing/phase0/`. What each of them reports
//! today, with the counts, is
//! `revision/phase0/results/REPORT.md`; no number is repeated here, because
//! a number in a doc comment ages.
//!
//! # Building and testing
//!
//! ```text
//! Z3LIB=$(.venv/bin/python -c "import z3,pathlib;\
//!   p=pathlib.Path(z3.__file__).parent;print(next(p.glob('**/libz3.so')).parent)")
//! PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 LIBRARY_PATH=$Z3LIB LD_LIBRARY_PATH=$Z3LIB \
//!   cargo test --lib --no-default-features --features halo2-backend phase0
//! ```
//!
//! The default feature set includes pyo3's `extension-module`, which does not
//! link a test binary, so `--no-default-features` is needed; `halo2-backend`
//! has to stay on because `src/prove/mock/tests.rs` imports
//! `crate::prove::halo2`, and that pulls in Z3, so `LIBRARY_PATH` and
//! `LD_LIBRARY_PATH` must name the directory holding `libz3.so` from the
//! `z3-solver` wheel or the test binary does not link. On a machine whose
//! Python is newer than pyo3 supports, export
//! `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` first, as `scripts/develop.sh`
//! does. Tests elsewhere in the crate (`ir_gen`, `optim`, `prove`) fail under
//! this feature set independently of this module, so filter to `phase0`.

pub mod desc;
pub mod export;
pub mod interval;
pub mod ops;
pub mod phi;
pub mod rules;
pub mod source;
pub mod term;
mod walk;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use crate::ast::ASTCircuit;
use crate::optim::predicates::formula::ContractTerm;
use crate::types::ZinniaType;

use desc::{Desc, KeyCursor, ValueKey};
use phi::Phi;
use rules::inputs::{add_scalar_requires, e_input, ContractSymbols};
use rules::RuleCtx;
use source::{Producer, ValueFrame, ValueStack};
use term::{BoolId, TermArena};

/// Why phase 0 refused a program.
///
/// The variants are the rejection reasons of `definition.md` §1 plus a
/// catch-all, so that a coverage report can histogram them
/// (`acceptance.md` §1). A rejection is a property of the program, never of
/// a run: a `while` loop is never rejected (it is unrolled to its cap) and a
/// solver timeout can never occur, because phase 0 calls no solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reject {
    /// Rank or dtype-family mismatch at a merge (§6.1).
    MergeMismatch { detail: String },
    /// A slice with a non-constant step (§5.4).
    NonConstantStep { detail: String },
    /// A reshape whose input size is not a term, or whose target has a
    /// non-constant extent next to `-1` (§5.6).
    ReshapeNotSupported { detail: String },
    /// A value-sized construct without a declared maximum (§5.9).
    ValueSizedWithoutMaximum { construct: String },
    /// A `range` whose bound is not a constant term (§6.2).
    NonConstantRangeBound { detail: String },
    /// Recursion deeper than `recursion_limit` (§5.10).
    RecursionTooDeep { limit: u32 },
    /// An unpacking with mismatched arity (§5.11).
    UnpackArity { expected: usize, found: usize },
    /// A `DynamicNDArray` input with runtime rank (§5.1).
    RuntimeRankInput { input: String },
    /// An operator outside §5 and `coverage.md`.
    UnsupportedOperator { name: String },
    /// A premise of a rule does not hold: the rank, axis, dtype or arity
    /// requirement §4 states as a premise, which is decided statically and
    /// so is a static error (§4, "premises on the input descriptors").
    ///
    /// `definition.md` §4 keeps this apart from a construct that has no rule
    /// at all, which is [`Reject::Unsupported`]; the coverage histogram
    /// keeps them apart for the same reason.
    Premise { rule: rules::Rule, detail: String },
    /// The unrolled program is larger than phase 0 will walk, or the record
    /// it has built is larger than phase 0 will hold.
    ///
    /// Not a reason of §1: unrolling a loop nest whose trip counts come from
    /// declared maxima can be quadratic in those maxima (§6.2 assumes the
    /// trip count is a compilation parameter), and a pass that never returns
    /// — or that is killed for its size — is worse than one that says why.
    /// `units` is the sum the budget bounds (§6.2). See "Additions" in
    /// `implementation.md`.
    BudgetExceeded { units: u64, budget: u64 },
    /// Anything else phase 0 does not handle yet.
    Unsupported { construct: String },
}

impl Reject {
    /// A stable short code, so a coverage report can histogram reasons.
    pub fn reason(&self) -> &'static str {
        match self {
            Reject::MergeMismatch { .. } => "merge_mismatch",
            Reject::NonConstantStep { .. } => "non_constant_step",
            Reject::ReshapeNotSupported { .. } => "reshape_not_supported",
            Reject::ValueSizedWithoutMaximum { .. } => "value_sized_without_maximum",
            Reject::NonConstantRangeBound { .. } => "non_constant_range_bound",
            Reject::RecursionTooDeep { .. } => "recursion_too_deep",
            Reject::UnpackArity { .. } => "unpack_arity",
            Reject::RuntimeRankInput { .. } => "runtime_rank_input",
            Reject::UnsupportedOperator { .. } => "unsupported_operator",
            Reject::Premise { .. } => "premise",
            Reject::BudgetExceeded { .. } => "budget_exceeded",
            Reject::Unsupported { .. } => "unsupported",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            Reject::MergeMismatch { detail }
            | Reject::NonConstantStep { detail }
            | Reject::ReshapeNotSupported { detail }
            | Reject::NonConstantRangeBound { detail } => detail.clone(),
            Reject::ValueSizedWithoutMaximum { construct } | Reject::Unsupported { construct } => {
                construct.clone()
            }
            Reject::RecursionTooDeep { limit } => format!("recursion_limit is {}", limit),
            Reject::UnpackArity { expected, found } => {
                format!("expected {} targets, found {}", expected, found)
            }
            Reject::RuntimeRankInput { input } => input.clone(),
            Reject::UnsupportedOperator { name } => name.clone(),
            Reject::Premise { rule, detail } => format!("{} {}", rule.name(), detail),
            Reject::BudgetExceeded { units, budget } => format!(
                "the unrolled program reached {} units of work and record, past the \
                 budget of {} (raise `phase0_work_budget`)",
                units, budget
            ),
        }
    }

    /// The form `analyze_shapes` returns to Python, so that the coverage
    /// harness sees a result rather than an exception.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"reason": self.reason(), "detail": self.detail()})
    }
}

impl std::fmt::Display for Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.reason(), self.detail())
    }
}

/// The compilation parameters phase 0 reads from the config JSON
/// (`coverage.md` §7). The rest of $S$ — the admitted input domain, the
/// declared maxima, the structural `while` caps — comes from the annotations
/// and the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phase0Config {
    pub loop_limit: u32,
    pub recursion_limit: u32,
    /// How much work the walker will do before giving up, and with it the
    /// largest number of iterations or elements it will materialise
    /// ([`interval::declared_maximum`]).
    ///
    /// §6.2 argues that unrolling is affordable because the trip count is a
    /// compilation parameter. When the trip count is instead a declared
    /// maximum, a loop nest is quadratic in that maximum and the pass does
    /// not finish, so the walker charges one unit per statement, per
    /// expression node, per unrolled iteration, and per descriptor leaf
    /// merged at a dynamic `if`, and stops here with
    /// [`Reject::BudgetExceeded`]. Raise it with the config key
    /// `phase0_work_budget`.
    pub work_budget: u64,
}

impl Default for Phase0Config {
    fn default() -> Self {
        Phase0Config {
            loop_limit: 1000,
            recursion_limit: 16,
            work_budget: 4_000_000,
        }
    }
}

impl Phase0Config {
    /// Read the same config JSON `compile_circuit` receives
    /// (`src/lib.rs`), with the same JSON defaults.
    pub fn from_json(config: &serde_json::Value) -> Self {
        let default = Phase0Config::default();
        Phase0Config {
            loop_limit: config["loop_limit"]
                .as_u64()
                .unwrap_or(default.loop_limit as u64) as u32,
            recursion_limit: config["recursion_limit"]
                .as_u64()
                .unwrap_or(default.recursion_limit as u64) as u32,
            // Not a key `compile_circuit` reads: phase 0 alone has a work
            // budget. `ZinniaCompiler._core_payloads` forwards it when the
            // caller sets it (`config.set("phase0_work_budget", n)`).
            work_budget: config["phase0_work_budget"]
                .as_u64()
                .unwrap_or(default.work_budget),
        }
    }
}

/// One entry of $\Delta$: a descriptor under its value key, with the source
/// name when the value has one (a circuit input, a bound name, the delivery)
/// and the [`Producer`] that made it.
///
/// $\Delta$ maps *every* value the walk keys, intermediates included
/// (§1), so most entries have no name.
#[derive(Debug, Clone)]
pub struct DescriptorEntry {
    pub key: ValueKey,
    pub name: Option<String>,
    pub desc: Desc,
    /// Where the value came from: the source graph of §3.3
    /// (`implementation.md` §5a).
    pub producer: Producer,
    /// How many checks $\Phi$ held when the entry was recorded, the value's
    /// own among them: the entry is made once its rules have run. It is what
    /// orders the values of $\Delta$ against the checks of $\Phi$
    /// (`implementation.md` §5a).
    pub checks: u32,
}

/// The result of phase 0: $(\Phi, \Delta, R)$ and the term arena every term
/// id refers to.
pub struct Phase0Result {
    pub terms: TermArena,
    pub phi: Phi,
    /// $\Delta$, in traversal order.
    pub descriptors: Vec<DescriptorEntry>,
    /// $R$, the delivery set: the returned values (§1).
    pub delivery: Vec<ValueKey>,
    pub config: Phase0Config,
    index: HashMap<ValueKey, usize>,
}

impl Phase0Result {
    pub fn descriptor(&self, key: &ValueKey) -> Option<&Desc> {
        self.entry(key).map(|entry| &entry.desc)
    }

    /// The whole entry under `key`: the descriptor together with the
    /// producer and the check watermark phase 1 reads
    /// (`revision/phase1/README.md` §1).
    pub fn entry(&self, key: &ValueKey) -> Option<&DescriptorEntry> {
        self.index.get(key).map(|&i| &self.descriptors[i])
    }
}

/// The mutable state of a pass: `implementation.md` §3's `State` without
/// `Env`, which the walker owns.
pub(crate) struct Analysis {
    pub terms: TermArena,
    pub phi: Phi,
    pub keys: KeyCursor,
    pub descriptors: Vec<DescriptorEntry>,
    /// $R$; the walker pushes the merged return value's key onto it.
    pub delivery: Vec<ValueKey>,
    /// The values under construction, which give each entry its producer and
    /// each check its subject (`implementation.md` §5a).
    values: ValueStack,
    config: Phase0Config,
    index: HashMap<ValueKey, usize>,
}

impl Analysis {
    pub fn new(config: Phase0Config) -> Self {
        Analysis {
            terms: TermArena::new(),
            phi: Phi::new(),
            keys: KeyCursor::new(),
            descriptors: Vec::new(),
            delivery: Vec::new(),
            values: ValueStack::default(),
            config,
            index: HashMap::new(),
        }
    }

    /// The rule context for the current path condition.
    ///
    /// A rule runs inside the value frame of the value it produces, which is
    /// what gives every check its subject, so this may only be called with a
    /// frame open ([`Analysis::open_value`]).
    pub fn ctx(&mut self, path: BoolId) -> RuleCtx<'_> {
        RuleCtx {
            terms: &mut self.terms,
            phi: &mut self.phi,
            keys: &mut self.keys,
            path,
            materialisable: self.config.work_budget as i128,
            values: &mut self.values,
            element: Vec::new(),
        }
    }

    /// Open the value frame of the value at `key` (`implementation.md` §5a).
    pub fn open_value(&mut self, key: &ValueKey) {
        self.values.open(key);
    }

    pub fn close_value(&mut self) -> ValueFrame {
        self.values.close()
    }

    /// Record that the value at `key` is an operand of the value being built.
    pub fn operand(&mut self, key: &ValueKey) {
        self.values.operand(key);
    }

    /// Record that the value being built is the value at `key` under a new
    /// key: a name reference, or an operator that returns its operand.
    pub fn alias(&mut self, key: &ValueKey) {
        self.values.alias(key);
    }

    /// Record one entry of $\Delta$.
    ///
    /// A repeated key is recorded again rather than overwriting, so that the
    /// uniqueness invariant of `implementation.md` §5a sees it; the lookup
    /// index keeps the first entry. Two values with one key can only come
    /// from a defect in the key discipline.
    pub fn record(&mut self, key: ValueKey, name: Option<String>, desc: Desc, producer: Producer) {
        self.index.entry(key.clone()).or_insert(self.descriptors.len());
        self.descriptors.push(DescriptorEntry {
            key,
            name,
            desc,
            producer,
            checks: self.phi.check_count(),
        });
    }

    /// The entry under `key`, when the walk has already recorded one.
    pub fn entry(&self, key: &ValueKey) -> Option<&DescriptorEntry> {
        self.index.get(key).map(|&at| &self.descriptors[at])
    }

    /// Open the frame of an *intermediate* of the value being built
    /// (`definition.md` §3.3) and return its key.
    ///
    /// The key is the value's own with the next step index, so it resolves
    /// to the same AST node (`crate::phase1::ast_index`) and a rule reads the
    /// operator's constant parameters off it. Until [`Analysis::close_step`]
    /// the intermediate is the value under construction: the rules that run
    /// are its, and it is the subject of the checks they record.
    pub fn open_step(&mut self) -> ValueKey {
        self.values.open_step()
    }

    /// Close an intermediate's frame: record its entry, with the rules it
    /// collected and the `operands` those rules read, and make it an operand
    /// of the value it belongs to.
    pub fn close_step(&mut self, desc: Desc, operands: Vec<ValueKey>) -> ValueKey {
        let frame = self.values.close();
        let key = frame.key().clone();
        let producer = Producer::of(&desc, frame.into_rules(), operands, None);
        self.record(key.clone(), None, desc, producer);
        self.operand(&key);
        key
    }

    /// The key Δ gives element `at` of the composite at `key`, when it gives
    /// it one (§3.3, "Elements of a composite"): the composite's key with the
    /// element's position as the step index, looked up through every entry
    /// that only relabels the composite.
    ///
    /// Phase 1 reads the same relation (`../phase1/build.rs`,
    /// `element_key`); here it is what names the operand an unpacking or an
    /// iteration binds.
    pub fn element_key(&self, key: &ValueKey, at: usize) -> Option<ValueKey> {
        let mut current = key.clone();
        for _ in 0..self.descriptors.len() {
            let candidate = current.intermediate(at as u32);
            if self.index.contains_key(&candidate) {
                return Some(candidate);
            }
            current = self
                .index
                .get(&current)
                .and_then(|&at| self.descriptors[at].producer.relabels())?
                .clone();
        }
        None
    }

    /// The key of the value `key` names, with every entry that only
    /// re-labels a value followed to the one that defines it: a use of a name
    /// and an `E-Copy` (§5.11). `Count` is keyed by this, so that two reads
    /// of one mask share one count variable (§5.5).
    pub fn defining_key(&self, key: &ValueKey) -> ValueKey {
        let mut current = key;
        while let Some(source) = self
            .index
            .get(current)
            .and_then(|&at| self.descriptors[at].producer.relabels())
        {
            current = source;
        }
        current.clone()
    }

    /// Called once the circuit body has been walked.
    pub fn finish(self) -> Phase0Result {
        debug_assert!(
            self.values.is_empty(),
            "every value frame the walk opened was closed"
        );
        Phase0Result {
            terms: self.terms,
            phi: self.phi,
            descriptors: self.descriptors,
            delivery: self.delivery,
            config: self.config,
            index: self.index,
        }
    }
}

/// Run phase 0 on one circuit.
///
/// `chips` and `externals` are the same maps `compile_circuit` receives:
/// chip name to `ASTChip` JSON, external name to its declaration. The
/// configuration, the contract facts and the input annotations come first —
/// they are the parameters $S$ of §1 — and then [`walk::Walker`] runs the
/// forward pass over the circuit body.
pub fn analyze(
    circuit: &ASTCircuit,
    chips: &HashMap<String, serde_json::Value>,
    externals: &HashMap<String, serde_json::Value>,
    config: &Phase0Config,
) -> Result<Phase0Result, Reject> {
    let mut analysis = Analysis::new(*config);
    let facts = scalar_precondition_terms(circuit);
    let mut symbols = ContractSymbols::default();
    symbols.harvest(&facts);
    let annotations = input_annotations(circuit)?;
    declare_shape_symbols(&mut analysis, &annotations, &mut symbols);
    let inputs = build_input_descriptors(&mut analysis, &annotations, &mut symbols)?;
    add_scalar_requires(
        &mut analysis.terms,
        &mut analysis.phi,
        &facts,
        &symbols,
    );
    let walker = walk::Walker::new(analysis, symbols, *config, chips, externals, inputs);
    walker.run(circuit)
}

/// The circuit's `scalar_requires`, parsed. An entry that does not
/// deserialize is dropped, exactly as `IRGenerator::scalar_precondition_terms`
/// drops it.
fn scalar_precondition_terms(circuit: &ASTCircuit) -> Vec<ContractTerm> {
    circuit
        .scalar_requires
        .iter()
        .filter_map(|fact| serde_json::from_value(fact.term.clone()).ok())
        .collect()
}

/// Parse every input annotation.
///
/// This is `IRGenerator::parse_dt_descriptor` (`src/ir_gen/helpers.rs:595`)
/// without its panic: an annotation phase 0 cannot read is a rejection, not
/// a crash. Phase 0 reads the *unresolved* annotation, `SymbolicNDArray`
/// included, because §5.1 turns the symbols into variables of $\Phi$ instead
/// of resolving them with a solver.
fn input_annotations(circuit: &ASTCircuit) -> Result<Vec<(String, ZinniaType)>, Reject> {
    circuit
        .inputs
        .iter()
        .map(|input| {
            let dt =
                serde_json::from_value::<ZinniaType>(input.annotation.dt.clone()).map_err(|e| {
                    Reject::Unsupported {
                        construct: format!(
                            "input `{}` has an unreadable annotation: {}",
                            input.name, e
                        ),
                    }
                })?;
            Ok((input.name.clone(), dt))
        })
        .collect()
}

/// Declare every shape symbol before any descriptor is built, so that a
/// scalar input named like a symbol binds to the same variable whatever the
/// order of the parameter list (§5.1: "shared by name across inputs").
fn declare_shape_symbols(
    analysis: &mut Analysis,
    annotations: &[(String, ZinniaType)],
    symbols: &mut ContractSymbols,
) {
    let Analysis { terms, phi, .. } = analysis;
    for (_, dt) in annotations {
        if let ZinniaType::SymbolicNDArray {
            shape, bindings, ..
        } = dt
        {
            for dim in shape {
                let (symbol, upper) = match dim {
                    crate::types::ShapeDim::Const(_) => continue,
                    crate::types::ShapeDim::Symbol(s) => (s.clone(), None),
                    crate::types::ShapeDim::Bounded { symbol, upper } => {
                        (symbol.clone(), Some(*upper as i128))
                    }
                };
                if bindings.contains_key(&symbol) {
                    continue;
                }
                symbols.declare(terms, phi, &symbol, upper);
            }
        }
    }
}

/// Apply the `E-Input-*` rules of §5.1 and record the descriptors in
/// $\Delta$.
///
/// Circuit inputs sit outside every statement, so their keys have an empty
/// statement path and the input's position as the expression index; no
/// expression key can collide with them, because an expression always lies
/// inside a statement.
fn build_input_descriptors(
    analysis: &mut Analysis,
    annotations: &[(String, ZinniaType)],
    symbols: &mut ContractSymbols,
) -> Result<Vec<(String, Desc)>, Reject> {
    let mut inputs = Vec::with_capacity(annotations.len());
    for (position, (name, dt)) in annotations.iter().enumerate() {
        let key = analysis.keys.at(position as u32);
        let tt = analysis.terms.tt();
        // The input's own key is the subject of every check `E-Input-*`
        // records (`implementation.md` §5a).
        analysis.open_value(&key);
        let desc = e_input(&mut analysis.ctx(tt), name, &key, dt, symbols);
        analysis.close_value();
        let desc = desc?;
        analysis.record(key.clone(), Some(name.clone()), desc.clone(), Producer::Input);
        // A composite parameter is a tuple or list of parameters, and each
        // element is a value of the circuit in its own right: §3.3 keys the
        // elements of a composite wherever an element has no other name, and
        // an element of an input has none. The producer is `Input` too — the
        // element is bound by the front end like any other parameter — which
        // is what lets phase 1 answer `a_min_val` in
        // `benchmarking/pythran_extrema` (`../phase1/rules/sources/input_scalar.md`).
        if let Desc::Composite(composite) = &desc {
            for (at, element) in composite.elems.iter().enumerate() {
                analysis.record(
                    key.intermediate(at as u32),
                    None,
                    element.clone(),
                    Producer::Input,
                );
            }
        }
        inputs.push((name.clone(), desc));
    }
    Ok(inputs)
}
