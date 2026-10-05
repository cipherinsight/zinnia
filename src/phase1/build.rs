//! The construction of `revision/phase1/README.md` §4.
//!
//! The order is §4's, literally: the CANFAIL sweep over the checks, then the
//! roots, then a depth-first expansion of every row, then `Fetch` and
//! `Enforce`. The state is the term arena, the provider table, the graph and
//! the oracle; a rule sees them through [`Builder`], which is also where the
//! things every rule needs live — the descriptor of a value, the AST node
//! its key names, the width policy of `rules/README.md` §6.1, and the
//! child-row call that carries the epoch filter.
//!
//! # The term arena
//!
//! [`Builder::terms`] starts as a clone of phase 0's arena and grows with
//! the terms phase 1 builds (§1.1). Every id below [`Builder::base_nodes`]
//! is a term phase 0 built; every id at or above it is one phase 1 built.
//! Phase 1 builds only through the arena's constructors, so the language
//! stays phase 0's and every question stays in the decidable fragment; it
//! never declares a variable and never adds a constraint to Φ.

use std::collections::{BTreeSet, HashMap};

use crate::ast::ASTCircuit;
use crate::phase0::desc::{Desc, Dtype, ScalarDesc, ScalarTerm, ValueKey};
use crate::phase0::interval::Interval;
use crate::phase0::phi::{ConstraintKind, VarOrigin};
use crate::phase0::source::Producer;
use crate::phase0::term::{BoolId, NodeView, TermArena, TermId, VarId};
use crate::phase0::{DescriptorEntry, Phase0Result};

use crate::phase1::ast_index::{AstIndex, AstNode, InputNode};
use crate::phase1::cost::Cost;
use crate::phase1::graph::{
    EnforceTarget, EpochFilter, Form, Graph, ImplId, ImplNode, RequirementId, SharingId, Target,
};
use crate::phase1::interface::{Batch, Capacity, Family, Layout, RanksOf};
use crate::phase1::omega::Omega;
use crate::phase1::oracle::answers::{AnswerId, Verdict};
use crate::phase1::oracle::layers;
use crate::phase1::oracle::questions::{Condition, Question};
use crate::phase1::oracle::{Limits, Oracle};
use crate::phase1::rules::{Candidate, RuleId, RULES};
use crate::phase1::table::{Row, RowEntry, RowRequest, Table};
use crate::phase1::Reject;

/// The state of one construction.
pub(crate) struct Builder<'a> {
    pub result: &'a Phase0Result,
    /// The term arena phase 1 asks and builds over (§1.1): phase 0's, cloned
    /// and extended. `result.terms` is left alone.
    pub terms: TermArena,
    /// `terms.node_count()` before phase 1 built anything: the id at which
    /// phase 1's own terms start, which is what the export writes out.
    pub base_nodes: usize,
    pub ast: AstIndex,
    pub omega: &'a Omega,
    pub cost: &'a dyn Cost,
    pub oracle: Oracle<'a>,
    pub graph: Graph,
    pub table: Table,
    /// Key to its entry of Δ, so a rule reaches a producer in one step.
    entries: HashMap<ValueKey, usize>,
    /// Arena node to the scalar value of Δ whose term it is
    /// (`rules/README.md` §3.2: the two rows coincide). Only phase 0's
    /// nodes are in it: a term phase 1 built is the term of no value.
    term_owner: HashMap<u32, ValueKey>,
    /// Input name to its key, for the shared shape symbols of
    /// `input_store.md`.
    inputs: HashMap<String, ValueKey>,
    /// Rows that were asked for and had no provider, for the rejection.
    pub missing: Vec<String>,
    /// One frame per row being filled: `Some(n)` once a port of one of its
    /// candidates asked for a row that was already being filled (§4,
    /// "Acyclicity by construction"). `n` numbers the fill, so the
    /// provisional row's requirement nodes are distinct from every other
    /// fill's.
    provisional: Vec<Option<u32>>,
    /// The next fill number.
    fills: u32,
}

impl<'a> Builder<'a> {
    pub fn new(
        result: &'a Phase0Result,
        circuit: &ASTCircuit,
        chips: &HashMap<String, serde_json::Value>,
        omega: &'a Omega,
        cost: &'a dyn Cost,
        limits: Limits,
    ) -> Builder<'a> {
        let mut entries = HashMap::new();
        let mut term_owner: HashMap<u32, ValueKey> = HashMap::new();
        let mut inputs = HashMap::new();
        for (at, entry) in result.descriptors.iter().enumerate() {
            entries.entry(entry.key.clone()).or_insert(at);
            if matches!(entry.producer, Producer::Input) {
                if let Some(name) = &entry.name {
                    inputs.entry(name.clone()).or_insert(entry.key.clone());
                }
            }
            // A relabelling entry — an alias, an `E-Copy`, a one-operand
            // `S-Assign` — has the rows of the value it names, so it owns no
            // term: making it the owner would send the value's own row back
            // to the alias and the alias's row back to the value.
            if entry.producer.relabels().is_some() {
                continue;
            }
            let Desc::Scalar(scalar) = &entry.desc else {
                continue;
            };
            let index = match scalar.term {
                // A literal is substituted at every use, so it owns no row:
                // `SCALAR_OP/const` answers it wherever it is asked.
                ScalarTerm::Int(t) if result.terms.constant_value(t).is_none() => t.index() as u32,
                ScalarTerm::Bool(b)
                    if !result.terms.is_true(b) && !result.terms.is_false(b) =>
                {
                    b.index() as u32
                }
                _ => continue,
            };
            // The first entry in Δ order owns the term, and an entry whose
            // ψ is `true` is preferred: every consumer can name it.
            let unconditional = result.terms.is_true(scalar.psi);
            match term_owner.get(&index) {
                Some(previous) => {
                    let previous_is_conditional = entries
                        .get(previous)
                        .and_then(|&at| result.descriptors[at].desc.psi())
                        .map(|psi| !result.terms.is_true(psi))
                        .unwrap_or(false);
                    if unconditional && previous_is_conditional {
                        term_owner.insert(index, entry.key.clone());
                    }
                }
                None => {
                    term_owner.insert(index, entry.key.clone());
                }
            }
        }
        Builder {
            oracle: Oracle::new(&result.terms, &result.phi, limits),
            terms: result.terms.clone(),
            base_nodes: result.terms.node_count(),
            ast: AstIndex::build(circuit, chips),
            result,
            omega,
            cost,
            graph: Graph::default(),
            table: Table::default(),
            entries,
            term_owner,
            inputs,
            missing: Vec::new(),
            provisional: Vec::new(),
            fills: 0,
        }
    }

    // ── Δ, Φ and P ───────────────────────────────────────────────────

    pub fn entry(&self, key: &ValueKey) -> Option<&DescriptorEntry> {
        self.entries.get(key).map(|&at| &self.result.descriptors[at])
    }

    pub fn scalar(&self, key: &ValueKey) -> Option<&ScalarDesc> {
        self.entry(key).and_then(|entry| entry.desc.as_scalar())
    }

    pub fn array(&self, key: &ValueKey) -> Option<&crate::phase0::desc::ArrayDesc> {
        self.entry(key).and_then(|entry| entry.desc.as_array())
    }

    pub fn node(&self, key: &ValueKey) -> Option<&AstNode> {
        self.ast.node(key)
    }

    pub fn input_node(&self, key: &ValueKey) -> Option<&InputNode> {
        self.ast.input(key)
    }

    pub fn input_key(&self, name: &str) -> Option<&ValueKey> {
        self.inputs.get(name)
    }

    /// The value that computes the term at this arena index, when one does
    /// (`rules/README.md` §3.2).
    pub fn owner_of_term(&self, index: u32) -> Option<&ValueKey> {
        self.term_owner.get(&index)
    }

    /// ψ of a value, the condition its providers satisfy their contract
    /// under (`rules/README.md` §3.5).
    pub fn psi(&self, key: &ValueKey) -> BoolId {
        self.entry(key)
            .and_then(|entry| entry.desc.psi())
            .unwrap_or_else(|| self.terms.tt())
    }

    /// The condition a child row is raised under (§4, "Conditions of child
    /// requirements"): the child's own ψ, so that every consumer of that
    /// child shares one row.
    ///
    /// §4's premise is that the consumer can name the child, so the
    /// condition `under` it reads the child under implies ψ. It is checked
    /// here and not assumed: syntactically, when every conjunct of ψ is a
    /// conjunct of `under`, and otherwise by `PROVE(under ⇒ ψ)`, which is
    /// the fallback `rules/README.md` §3.5 names. A child that neither
    /// settles is a value the consumer cannot name — a defect of the record
    /// rather than a missing rule — so phase 1 rejects and names both
    /// conditions.
    ///
    /// `under` is one Boolean term, the arm condition included: `SELECT`
    /// reads its two sides under the terms `ψ_v ∧ b` and `ψ_v ∧ ¬b`
    /// (`select.md`, `correctness_theorem`), which [`conjunction`] builds.
    pub fn child_condition(
        &mut self,
        candidate: &mut Candidate,
        under: Condition,
        psi: BoolId,
        named: &str,
    ) -> Result<BoolId, Reject> {
        if self.terms.is_true(psi) {
            return Ok(psi);
        }
        if layers::conjuncts(&self.terms, psi)
            .is_subset(&layers::conjuncts(&self.terms, under))
        {
            return Ok(psi);
        }
        // Epoch 0: the ranges, the preconditions and the declarations, with
        // no check assumed. A proof there holds at every epoch, and a ψ that
        // needs a check to follow is not something a consumer may rely on.
        if self.ask_for(candidate, Question::Prove(psi), 0, under) == Verdict::Proved {
            return Ok(psi);
        }
        Err(Reject::ChildCondition {
            value: named.to_string(),
            under: self.render_bool(under),
            psi: self.render_bool(psi),
        })
    }

    /// [`Builder::child_condition`] for an ordinary port, read under the
    /// row's own `g`.
    pub fn child_under_g(
        &mut self,
        candidate: &mut Candidate,
        request: &RowRequest,
        psi: BoolId,
        named: &str,
    ) -> Result<BoolId, Reject> {
        self.child_condition(candidate, request.condition, psi, named)
    }

    /// The condition a row's questions are asked under, which is the row's
    /// own formula: a term already, since phase 1 builds terms.
    pub fn under(&self, formula: BoolId) -> Condition {
        formula
    }

    /// The arm condition `ψ_v ∧ b` or `ψ_v ∧ ¬b` of `select.md`, built
    /// through [`conjunction`] so that the conjunct set fixes the term id.
    pub fn under_arm(&mut self, formula: BoolId, arm: BoolId, negated: bool) -> Condition {
        let literal = if negated { self.terms.not(arm) } else { arm };
        conjunction(&mut self.terms, &[formula, literal])
    }

    /// One Boolean term as text, for a rejection message.
    pub fn render_bool(&self, formula: BoolId) -> String {
        self.terms.pretty_bool(formula).chars().take(160).collect()
    }

    /// The descriptor an element path names inside a value (§3.4, §3.3): the
    /// value's own descriptor for the empty path, otherwise the element the
    /// index sequence reaches.
    pub fn element_of(&self, key: &ValueKey, path: &[usize]) -> Option<&Desc> {
        let mut desc = &self.entry(key)?.desc;
        for &index in path {
            desc = desc.as_composite()?.elems.get(index)?;
        }
        Some(desc)
    }

    /// The row an element of a value is named by: the value itself for the
    /// empty path, otherwise the element's own term, because Δ keys the
    /// composite and not its elements (`rules/README.md` §2).
    ///
    /// Two names come before the term. Δ keys the elements of a merged
    /// composite, of the composite a list method rebound and of one a value
    /// passed through (`../phase0/definition.md` §3.3), and such an element
    /// is named by its own key — which is the only name it has when its
    /// descriptor carries no term, a `Float` scalar or an array. Failing
    /// that, an element of a literal `E-Composite` whose operands stand one
    /// per element is named by the operand it was built from, which is what
    /// gives a delivered `(a, b)` of Float scalars a row per element.
    pub fn element_target(&self, key: &ValueKey, path: &[usize]) -> Option<Target> {
        if path.is_empty() {
            return Some(Target::Value(key.clone()));
        }
        // Δ keys the elements of a merged composite (`../phase0/definition.md`
        // §3.3), and an element that has a key of its own is named by it:
        // that is the only name an element whose descriptor carries no term
        // has, and where the element also carries one the two rows coincide
        // through the term's owner (`rules/README.md` §3.2).
        if let Some(element) = self.element_key(key, path) {
            return Some(Target::Value(element));
        }
        let by_term = match self.element_of(key, path)? {
            Desc::Scalar(scalar) => match scalar.term {
                ScalarTerm::Int(t) => Some(Target::Term(t)),
                ScalarTerm::Bool(b) => Some(Target::BoolTerm(b)),
                ScalarTerm::Opaque => None,
            },
            _ => None,
        };
        by_term.or_else(|| self.composite_operand(key, path).map(Target::Value))
    }

    /// The key Δ gives one element of a composite, when it gives it one
    /// (`../phase0/definition.md` §3.3): the composite's key with the
    /// element's position as the step index, looked up through every entry
    /// that only relabels the composite — a binding, an `E-Copy`, the
    /// delivery of a single `return`.
    fn element_key(&self, key: &ValueKey, path: &[usize]) -> Option<ValueKey> {
        let [index] = path else {
            return None;
        };
        let mut current = key.clone();
        loop {
            let candidate = current.intermediate(*index as u32);
            if self.entry(&candidate).is_some() {
                return Some(candidate);
            }
            current = self.entry(&current)?.producer.relabels()?.clone();
        }
    }

    /// The composite an element key names and the element's position, for a
    /// rule that is asked for the element's row and needs the merge the
    /// composite records (`../phase0/definition.md` §3.3).
    pub fn element_of_key(&self, key: &ValueKey) -> Option<(ValueKey, usize)> {
        if key.step == 0 {
            return None;
        }
        let composite = ValueKey { step: 0, ..key.clone() };
        self.entry(&composite)?.desc.as_composite()?;
        Some((composite, key.step as usize - 1))
    }

    /// The value one element of a composite literal came from: the operand
    /// at that index, level by level down the path.
    ///
    /// Only for a composite whose producer is an `E-Composite` with exactly
    /// one operand per element — a tuple or list written in the source — and
    /// only when the operand's descriptor is the element's kind, so an
    /// operand list a starred element flattened is refused rather than
    /// misread.
    fn composite_operand(&self, key: &ValueKey, path: &[usize]) -> Option<ValueKey> {
        let mut current = key.clone();
        for &index in path {
            // An alias — a binding, an `E-Copy` — has the rows of the value
            // it names (`rules/README.md` §2), and the producer that lists
            // the elements is that value's.
            while let Some(source) = self.entry(&current)?.producer.relabels() {
                current = source.clone();
            }
            let entry = self.entry(&current)?;
            let composite = entry.desc.as_composite()?;
            let Producer::Applied { rules, operands, condition: None } = &entry.producer else {
                return None;
            };
            if !rules.contains(&crate::phase0::rules::Rule::EComposite) {
                return None;
            }
            if operands.len() != composite.elems.len() {
                return None;
            }
            let operand = operands.get(index)?.clone();
            let element = composite.elems.get(index)?;
            if self.entry(&operand)?.desc.kind_str() != element.kind_str() {
                return None;
            }
            current = operand;
        }
        Some(current)
    }

    /// ψ of an element of a value: a composite has none of its own, so the
    /// condition is the element's (`../../phase0/definition.md` §3.3).
    pub fn element_psi(&self, key: &ValueKey, path: &[usize]) -> BoolId {
        self.element_of(key, path)
            .and_then(|desc| desc.psi())
            .unwrap_or_else(|| self.terms.tt())
    }

    /// ψ the providers of a row satisfy their contract under
    /// (`rules/README.md` §3.5).
    ///
    /// For a value row it is the value's own ψ. A term row is the row of the
    /// value that owns the term (§3.2), so it is that value's ψ, which is
    /// what makes one occurrence of a name inside a branch share the row of
    /// the value it names. A term no value owns — the fresh variable of a
    /// merged composite element — keeps the `fallback` the caller read off
    /// the descriptor that introduced it.
    pub fn target_psi(&self, target: &Target, fallback: BoolId) -> BoolId {
        let index = match target {
            Target::Value(key) => return self.psi(key),
            Target::Term(t) => t.index() as u32,
            Target::BoolTerm(b) => b.index() as u32,
        };
        match self.owner_of_term(index) {
            Some(owner) => self.psi(owner),
            None => fallback,
        }
    }

    /// The value the variable at this arena index was introduced by, as a
    /// (key, element path) pair (`rules/README.md` §2): the composite a
    /// merged element came out of, and the path down to that element.
    pub fn variable_origin(&self, index: u32) -> Option<(ValueKey, Vec<usize>)> {
        let variable = self.variable_at(index)?;
        let info = self.result.phi.var_info(variable);
        let key = match &info.origin {
            VarOrigin::DataScalar { value } | VarOrigin::DataBool { value } => value.clone(),
            _ => return None,
        };
        Some((key, info.element_path.clone()))
    }

    /// The variable a term node is, at either sort.
    pub fn variable_at(&self, index: u32) -> Option<VarId> {
        match self.terms.node_at(index as usize) {
            NodeView::Var(variable) | NodeView::BVar(variable) => Some(variable),
            _ => None,
        }
    }

    /// The first input in Δ order whose descriptor carries this term as a
    /// dimension (`rules/README.md` §2: "the first parameter in Δ order that
    /// binds a shape symbol shared by several inputs").
    pub fn first_input_with_dim(&self, term: TermId) -> Option<ValueKey> {
        self.result
            .descriptors
            .iter()
            .find(|entry| {
                matches!(entry.producer, Producer::Input)
                    && entry
                        .desc
                        .as_array()
                        .map(|array| array.dims.contains(&term))
                        .unwrap_or(false)
            })
            .map(|entry| entry.key.clone())
    }

    /// The value another expression of the same statement computed
    /// (`ast_index::SubscriptEntry`): the same statement path, iteration
    /// counters and call sites with another expression index.
    ///
    /// This is how a rule names a value the AST node points at but the
    /// producer chain does not distinguish — the `lo` of a slice entry, the
    /// index array of a subscript — and it is exact, because the key scheme
    /// is (statement path, iterations, calls, expression index) and phase 1's
    /// index replays phase 0's numbering.
    pub fn sibling(&self, key: &ValueKey, expr_index: u32) -> Option<&DescriptorEntry> {
        self.entry(&ValueKey { expr_index, ..key.clone() })
    }

    /// The check watermark of a value: the default epoch of a node that
    /// represents it (`rules/README.md` §3.7).
    pub fn watermark(&self, key: &ValueKey) -> u32 {
        self.entry(key).map(|entry| entry.checks).unwrap_or(0)
    }

    pub fn interval(&self, term: TermId) -> Interval {
        layers::interval(&self.terms, &self.result.phi, term)
    }

    /// The dtype width `w_τ` of §6.1.
    pub fn dtype_width(&self, dtype: Dtype) -> u32 {
        match dtype {
            Dtype::Bool => 1,
            Dtype::Idx | Dtype::I64 => 64,
            _ => 0,
        }
    }

    // ── The oracle ───────────────────────────────────────────────────

    /// Ask a question and keep its id on the candidate that asked it (§2:
    /// every question a node asked is in A).
    pub fn ask_for(
        &mut self,
        candidate: &mut Candidate,
        question: Question,
        epoch: u32,
        condition: Condition,
    ) -> Verdict {
        let at = self.oracle.ask(&self.terms, question, epoch, condition);
        candidate.answers.push(at);
        self.verdict_of(at)
    }

    pub fn verdict_of(&self, at: AnswerId) -> Verdict {
        self.oracle
            .answers
            .get(at)
            .map(|answer| answer.verdict.clone())
            .unwrap_or(Verdict::NotProved)
    }

    // ── Capacities ───────────────────────────────────────────────────

    /// The capacity box of a value: per axis, the CONST extent or the MAX
    /// over Φ_e ∧ g, with the later-epoch variants of `rules/README.md` §6.2.
    ///
    /// Every rule that allocates a capacity through MAX calls this, so the
    /// variant mechanism is written once: a rule whose capacity comes from a
    /// MAX "may instantiate a second node at a later epoch e′ when MAX over
    /// Φ_{e′} is smaller". MAX is monotone in the epoch — a larger Φ_e can
    /// only shrink the maximum — so the smallest capacity is the one at the
    /// last check of Φ, and that is the variant offered; the default-epoch
    /// box stays beside it, because below `Enforce(k)` the epoch filter
    /// admits only the node that assumed no check at or after k. `Ω.epoch_variants`
    /// caps how many variants a call may offer; intermediate positions are
    /// not tried.
    pub fn capacity_box(
        &mut self,
        candidate: &mut Candidate,
        dims: &[TermId],
        epoch: u32,
        under: Condition,
    ) -> Vec<CapacityBox> {
        let Some(base) = self.capacity_box_at(candidate, dims, epoch, under) else {
            return Vec::new();
        };
        let top = self.oracle.layers.check_count();
        if self.omega.epoch_variants == 0 || top <= epoch {
            return vec![base];
        }
        let Some(tight) = self.capacity_box_at(candidate, dims, top, under) else {
            return vec![base];
        };
        if tight.axes == base.axes {
            return vec![base];
        }
        vec![base, tight]
    }

    /// One capacity box, at one epoch.
    ///
    /// `None` when an axis has no bound at all, and when the box Ω admits is
    /// smaller than the one the answers ask for
    /// ([`Omega::max_capacity`]): a capacity is a resource parameter, and a
    /// rule may not allocate one outside the admitted values.
    fn capacity_box_at(
        &mut self,
        candidate: &mut Candidate,
        dims: &[TermId],
        epoch: u32,
        under: Condition,
    ) -> Option<CapacityBox> {
        let mut axes = Vec::with_capacity(dims.len());
        let mut constant = Vec::with_capacity(dims.len());
        for &dimension in dims {
            let answer = self.ask_for(candidate, Question::Const(dimension), epoch, under);
            if let Verdict::Constant(value) = answer {
                axes.push(value.max(0));
                constant.push(true);
                continue;
            }
            let bound = self.ask_for(candidate, Question::Max(dimension), epoch, under);
            let Verdict::Bound(capacity) = bound else {
                return None;
            };
            axes.push(capacity.max(0));
            constant.push(false);
        }
        let admitted = self.omega.max_capacity;
        let slots: Capacity = axes.iter().product();
        if axes.iter().any(|&axis| axis > admitted) || slots > admitted {
            let note = format!(
                "a capacity of {} slots is outside the values Ω admits ({}), so \
                 no rule allocates it",
                slots, admitted
            );
            if !self.missing.contains(&note) {
                self.missing.push(note);
            }
            return None;
        }
        Some(CapacityBox { epoch, axes, constant })
    }

    /// Whether two Store-family bundles are aligned (`rules/README.md`
    /// §4.1).
    ///
    /// Alignment is decided by construction except for the per-axis
    /// "PROVE(dims equal) under g" the definition asks for, which is this
    /// method's own question. Two axes that carry one hash-consed term are
    /// equal without a question, because a term id is an identity (§2).
    pub fn aligned(
        &mut self,
        candidate: &mut Candidate,
        left: &Layout,
        right: &Layout,
        left_dims: &[TermId],
        right_dims: &[TermId],
        epoch: u32,
        under: Condition,
    ) -> bool {
        if left_dims.len() != right_dims.len() {
            return false;
        }
        let pairs: Vec<(TermId, TermId)> = left_dims
            .iter()
            .copied()
            .zip(right_dims.iter().copied())
            .collect();
        let mut prove = || {
            pairs.iter().all(|&(a, b)| {
                // Two axes that carry one hash-consed term are equal without
                // a question; otherwise the claim is the built `a = b`.
                a == b || {
                    let claim = self.terms.eq(a, b);
                    self.ask_for(candidate, Question::Prove(claim), epoch, under)
                        == Verdict::Proved
                }
            })
        };
        crate::phase1::interface::aligned(left, right, &mut prove)
    }

    /// PROVE of one claim, through the oracle (`rules/README.md` §6.3).
    pub fn prove(
        &mut self,
        candidate: &mut Candidate,
        claim: BoolId,
        epoch: u32,
        under: Condition,
    ) -> bool {
        self.ask_for(candidate, Question::Prove(claim), epoch, under) == Verdict::Proved
    }

    /// `PROVE(a = b)` on the built equality (`rules/README.md` §6.3): the
    /// form alignment, a full reduced axis and a full contracted extent ask.
    pub fn prove_eq(
        &mut self,
        candidate: &mut Candidate,
        a: TermId,
        b: TermId,
        epoch: u32,
        under: Condition,
    ) -> bool {
        let claim = self.terms.eq(a, b);
        self.prove(candidate, claim, epoch, under)
    }

    /// `PROVE(a ≤ b)` on the built inequality: non-emptiness, a capacity
    /// covering an extent, an address in range.
    pub fn prove_le(
        &mut self,
        candidate: &mut Candidate,
        a: TermId,
        b: TermId,
        epoch: u32,
        under: Condition,
    ) -> bool {
        let claim = self.terms.le(a, b);
        self.prove(candidate, claim, epoch, under)
    }

    /// The literal `n` as a term, for a comparison against a capacity or a
    /// grid width.
    pub fn literal(&mut self, n: i128) -> TermId {
        self.terms.constant(n)
    }

    // ── Rows ─────────────────────────────────────────────────────────

    /// Fill the row a port asks for (§4, "Expansion").
    ///
    /// When the row is already being filled, the answer is the cut of §4,
    /// "Acyclicity by construction": an empty row, so the candidate whose
    /// port asked for it is dropped, and a mark on the row now being filled,
    /// which makes it provisional. The cut is per candidate — the other
    /// candidates of this row are built normally.
    pub fn child(
        &mut self,
        request: &RowRequest,
        target: Target,
        family: Family,
        condition: BoolId,
    ) -> Result<Row, Reject> {
        let epoch = self.epoch_for(&target, &request.filter);
        let child = request.child(target, family, condition, epoch);
        let row = self.row(&child)?;
        if row.cut {
            self.mark_provisional();
        }
        Ok(row)
    }

    /// Mark the row being filled as provisional and give the fill its
    /// number, once, so that every requirement node this fill creates is
    /// distinct from every other fill's.
    fn mark_provisional(&mut self) {
        let next = self.fills;
        if let Some(frame) = self.provisional.last_mut() {
            if frame.is_none() {
                *frame = Some(next);
                self.fills = next + 1;
            }
        }
    }

    /// The epoch a row's nodes ask their questions at: the value's check
    /// watermark, capped by the epoch filter the row carries
    /// (`rules/README.md` §6.2).
    pub fn epoch_for(&self, target: &Target, filter: &EpochFilter) -> u32 {
        let default = match target {
            Target::Value(key) => self.watermark(key),
            // A term is no value, so it has no watermark of its own; it
            // assumes every check its variables' values assumed.
            Target::Term(t) => self.term_epoch(t.index() as u32),
            Target::BoolTerm(b) => self.term_epoch(b.index() as u32),
        };
        match filter.epoch_below {
            Some(position) => default.min(position.saturating_sub(1)),
            None => default,
        }
    }

    fn term_epoch(&self, index: u32) -> u32 {
        match self.term_owner.get(&index) {
            Some(key) => self.watermark(key),
            None => 0,
        }
    }

    /// The child row of a read step, keyed by the batch of the row being
    /// filled: `Coords[Q, η]` and `ReadReq[o, Q, η]` are per η, the batch and
    /// the condition of the `Fetch` (`rules/README.md` §4.4).
    pub fn child_in_batch(
        &mut self,
        request: &RowRequest,
        target: Target,
        family: Family,
        condition: BoolId,
    ) -> Result<Row, Reject> {
        let epoch = self.epoch_for(&target, &request.filter);
        let child = RowRequest {
            batch: request.batch.clone(),
            ..request.child(target, family, condition, epoch)
        };
        let row = self.row(&child)?;
        if row.cut {
            self.mark_provisional();
        }
        Ok(row)
    }

    /// The row an `IndexNorm` port asks for: the normalised position of one
    /// index on one axis (`rules/README.md` §4.3).
    pub fn index_norm_port(
        &mut self,
        request: &RowRequest,
        index: Target,
        extent: TermId,
        condition: BoolId,
    ) -> Result<Row, Reject> {
        let epoch = self.epoch_for(&index, &request.filter);
        let keyed = request.child_index_norm(index, extent, condition, epoch);
        let row = self.row(&keyed)?;
        if row.cut {
            self.mark_provisional();
        }
        Ok(row)
    }

    /// The row a `Groups` port asks for: the fibers of a value over one axis
    /// set, under the value's own ψ (`rules/README.md` §4.5).
    pub fn groups_port(
        &mut self,
        candidate: &mut Candidate,
        request: &RowRequest,
        value: &ValueKey,
        axes: Vec<usize>,
    ) -> Result<Row, Reject> {
        let psi = self.psi(value);
        let condition = self.child_under_g(candidate, request, psi, &value.encode())?;
        let target = Target::Value(value.clone());
        let epoch = self.epoch_for(&target, &request.filter);
        let keyed = request.child_groups(target, axes, condition, epoch);
        let row = self.row(&keyed)?;
        if row.cut {
            self.mark_provisional();
        }
        Ok(row)
    }

    /// `Fetch(v, batch, g)` (§4 step 4).
    ///
    /// The row is the `Values` family of v under that batch, so every rule
    /// whose `answers` names `Fetch` fills it: the three read mechanisms of
    /// the reading group over each provider of v, and the rules that
    /// recompute v at the batch instead (`ELEMENTWISE_BATCH`). The one row
    /// that is no read is the scalar identity of `rules/README.md` §3.2,
    /// "for a scalar value u, `Represent(u, Wire[τ])` and `Fetch(u, [()], g)`
    /// are one requirement".
    pub fn fetch(
        &mut self,
        request: &RowRequest,
        value: &ValueKey,
        batch: &Batch,
        condition: BoolId,
    ) -> Result<Row, Reject> {
        let scalar = self.scalar(value).is_some();
        if scalar && matches!(batch, Batch::Scalar) {
            return self.child(request, Target::Value(value.clone()), Family::Wire, condition);
        }
        // Two `Fetch` rows over one value with different batches are two
        // rows, so the batch is part of the key the row is memoised under.
        let target = Target::Value(value.clone());
        let epoch = self.epoch_for(&target, &request.filter);
        let keyed = request.child_fetch(target, batch.clone(), condition, epoch);
        let row = self.row(&keyed)?;
        if row.cut {
            self.mark_provisional();
        }
        Ok(row)
    }

    /// [`Builder::fetch`] under the value's own ψ, with §4's premise checked
    /// against the row's `g` — the form every consumer of a whole value uses.
    pub fn fetch_port(
        &mut self,
        candidate: &mut Candidate,
        request: &RowRequest,
        value: &ValueKey,
        batch: &Batch,
    ) -> Result<Row, Reject> {
        let psi = self.psi(value);
        let condition = self.child_under_g(candidate, request, psi, &value.encode())?;
        self.fetch(request, value, batch, condition)
    }

    /// Fill one row (§4, "Expansion").
    pub fn row(&mut self, request: &RowRequest) -> Result<Row, Reject> {
        let key = request.key();
        if let Some(existing) = self.table.get(&key) {
            return Ok(existing.clone());
        }
        if let Some(row) = self.weaker_row(request) {
            return Ok(row);
        }
        if self.table.is_filling(&key) {
            // The row is its own ancestor, so the candidate that asked for
            // it is the one that would close the cycle. It is dropped and
            // named, and the row that holds it becomes provisional.
            let note = format!(
                "the row of {} for {} is being filled already, so a candidate \
                 that reads it to build it is dropped",
                request.target.render(),
                request.family.as_str()
            );
            return Ok(Row { cut: true, notes: vec![note], ..Row::default() });
        }
        // Asked before the row is opened, because it asks for the same
        // target and family with the filter removed.
        if let Some(row) = self.without_epoch_filter(request)? {
            if !row.provisional {
                self.table.insert(key, row.clone());
            }
            return Ok(row);
        }
        // The row is open from here on, so a delegation chain that would
        // come back to it stops instead of recurring.
        self.table.open(key.clone());
        self.provisional.push(None);
        let outcome = match self.delegation(request) {
            Some(delegated) => self.row(&delegated),
            None => self.fill(request),
        };
        let provisional = self.provisional.pop().flatten();
        self.table.close();
        let mut row = outcome?;
        row.provisional |= provisional.is_some();
        // A provisional row is missing the candidates the cut dropped, so it
        // belongs to the consumer that asked for it alone: the memo keeps
        // the unrestricted fill a later consumer will make.
        if !row.provisional {
            self.table.insert(key, row.clone());
        }
        Ok(row)
    }

    /// The unrestricted row, when the epoch filter of `rules/README.md` §6.2
    /// removes nothing from it.
    ///
    /// Below `Enforce(k)` every row carries "epoch < position(k)". Where
    /// every provider already satisfies it — which is the usual case,
    /// because a value's default epoch is its own check watermark — the
    /// filtered row *is* the unrestricted one, and returning it keeps one
    /// requirement node for the wire the check and the rest of the program
    /// share. Where the filter does remove a provider, the filtered row is
    /// its own row with its own requirement nodes, so that phase 2 cannot
    /// select the excluded provider for the check's wire.
    fn without_epoch_filter(&mut self, request: &RowRequest) -> Result<Option<Row>, Reject> {
        let Some(position) = request.filter.epoch_below else {
            return Ok(None);
        };
        let mut unrestricted = request.clone();
        unrestricted.filter.epoch_below = None;
        unrestricted.epoch = self.epoch_for(&request.target, &unrestricted.filter);
        let row = self.row(&unrestricted)?;
        if row.entries.iter().all(|entry| entry.epoch < position) {
            return Ok(Some(row));
        }
        Ok(None)
    }

    /// A row of the same key under a weaker condition answers this one
    /// (§4, "Conditions of child requirements"): syntactically, when every
    /// conjunct of the row's condition is a conjunct of `g`.
    fn weaker_row(&mut self, request: &RowRequest) -> Option<Row> {
        let key = request.key();
        let mine = layers::conjuncts(&self.terms, request.condition);
        let terms = &self.terms;
        let found = self.table.weaker(&key, &mine, |other| {
            layers::conjuncts(terms, other).is_subset(&mine)
        })?;
        Some(found.1.clone())
    }

    /// The row a request is answered by instead of itself.
    ///
    /// Three cases, all from `rules/README.md` §2 and §3.2. An `Alias(u)`
    /// entry has the rows of `u`: an alias has no node of its own. The row
    /// of a term that is the term of a scalar value of Δ is that value's
    /// row. And two scalar values of Δ that carry one hash-consed term are
    /// one row — §3.2's "the row of a scalar value of Δ and the row of its
    /// term are one row, registered under both keys", read through the term
    /// they share.
    ///
    /// Each is skipped when the row it would delegate to is the one being
    /// filled, which is the `LEN_SHAPE/term` case: inside a value's own row
    /// the term has to be compiled, not looked up.
    fn delegation(&self, request: &RowRequest) -> Option<RowRequest> {
        match &request.target {
            Target::Value(key) => self
                .alias_delegation(request, key)
                .or_else(|| self.composite_index_delegation(request, key))
                .or_else(|| self.read_delegation(request, key))
                .or_else(|| self.shared_term_delegation(request, key))
                .or_else(|| self.structured_term_delegation(request, key)),
            Target::Term(t) => self.term_delegation(request, t.index() as u32),
            Target::BoolTerm(b) => self.term_delegation(request, b.index() as u32),
        }
    }

    /// An `Alias(u)` entry has the rows of u (`rules/README.md` §2).
    ///
    /// The delegation holds only where the two descriptors agree. A binding
    /// that unpacks one element of a composite — `vals, idx = np.unique(a,
    /// return_inverse=True)` — is recorded as a single-operand `S-Assign`
    /// over the composite, which [`Producer::relabels`] reports as a
    /// relabelling although the descriptor is the *element*'s and not the
    /// composite's. Sending that row to the composite's would ask for a
    /// rank-1 array in the row of a Composite, which no rule fills; the
    /// value is named by the rule that built the composite instead
    /// (`ordering/unique.md`, `source_pattern`).
    fn alias_delegation(&self, request: &RowRequest, key: &ValueKey) -> Option<RowRequest> {
        let source = self.entry(key)?.producer.relabels()?;
        if self.entry(key)?.desc.kind_str() != self.entry(source)?.desc.kind_str() {
            return None;
        }
        self.redirect(request, Target::Value(source.clone()), request.family)
    }

    /// A value that indexes a composite at a literal position has the rows
    /// of that element (`rules/README.md` §2).
    ///
    /// `np.nonzero(m)[0]` and `interval[0]` are recorded as
    /// `Applied(E-Composite-Index, [L, k])` with L a Composite of Δ and k a
    /// literal, and the value's descriptor *is* the element's: it computes
    /// nothing, so it has no node of its own and every row it is asked for
    /// is the element's. Like [`Builder::alias_delegation`], the delegation
    /// holds only where the two descriptors agree in kind and where the
    /// element has a name of its own (`rules/README.md` §2, read by
    /// [`Builder::element_target`]).
    fn composite_index_delegation(
        &self,
        request: &RowRequest,
        key: &ValueKey,
    ) -> Option<RowRequest> {
        let entry = self.entry(key)?;
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return None;
        };
        if rules.as_slice() != [crate::phase0::rules::Rule::ECompositeIndex] {
            return None;
        }
        let composite = operands.first()?;
        let position = self
            .scalar(operands.get(1)?)
            .and_then(|scalar| scalar.term.as_int())
            .and_then(|term| self.terms.constant_value(term))?;
        let position = usize::try_from(position).ok()?;
        if self.element_of(composite, &[position])?.kind_str() != entry.desc.kind_str() {
            return None;
        }
        let target = self.element_target(composite, &[position]).or_else(|| {
            // A one-element tuple whose row a rule answers on the composite's
            // own key — `np.nonzero(m)`, whose `NONZERO` node is attached to
            // the tuple — gives its element no name of its own, and the
            // composite's row *is* the element's row, because the composite
            // holds nothing else.
            // Not for `Export`, which reads the descriptor of the row's
            // target and would find a Composite where the delivered value is
            // an array: the delivery keeps the index's own key and its ports
            // are the rows that do delegate.
            if request.family == Family::Export {
                return None;
            }
            let elements = self.entry(composite)?.desc.as_composite()?.elems.len();
            (elements == 1).then(|| Target::Value(composite.clone()))
        })?;
        self.redirect(request, target, request.family)
    }

    /// A scalar that is one element of an array is the one-entry `Fetch` of
    /// `rules/README.md` §3.2, read by the reading group: "for a scalar value
    /// u, `Represent(u, Wire[τ])` and `Fetch(u, [()], g)` are one
    /// requirement", and for an `E-Elem` the second is the row with a
    /// provider.
    fn read_delegation(&self, request: &RowRequest, key: &ValueKey) -> Option<RowRequest> {
        if request.family != Family::Wire {
            return None;
        }
        self.scalar(key)?;
        let Producer::Applied { rules, .. } = &self.entry(key)?.producer else {
            return None;
        };
        if !rules.iter().any(|rule| {
            matches!(
                rule,
                crate::phase0::rules::Rule::EElem
                    | crate::phase0::rules::Rule::EIndex
                    | crate::phase0::rules::Rule::ETake
                    | crate::phase0::rules::Rule::ETakeBroadcast
            )
        }) {
            return None;
        }
        if self.table.is_filling_target(&request.target, Family::Values) {
            return None;
        }
        Some(RowRequest {
            family: Family::Values,
            batch: Some(Batch::Scalar),
            ..request.clone()
        })
    }

    /// The value that owns the hash-consed term this scalar carries, when it
    /// is another value.
    fn shared_term_delegation(
        &self,
        request: &RowRequest,
        key: &ValueKey,
    ) -> Option<RowRequest> {
        if request.family != Family::Wire {
            return None;
        }
        let scalar = self.entry(key)?.desc.as_scalar()?;
        let index = match scalar.term {
            ScalarTerm::Int(t) => t.index() as u32,
            ScalarTerm::Bool(b) => b.index() as u32,
            ScalarTerm::Opaque => return None,
        };
        let owner = self.term_owner.get(&index)?;
        if owner == key {
            return None;
        }
        self.redirect(request, Target::Value(owner.clone()), Family::Wire)
    }

    /// A scalar whose own term is an expression rather than a variable: its
    /// wire *is* that term's wire, so the row delegates to the term and
    /// `TERM_TO_WIRES` compiles it node by node.
    ///
    /// The value owns the term, so [`Builder::term_delegation`] would send
    /// the term's row straight back here; it does not, because this row is
    /// open by the time it is asked (§4, "Acyclicity by construction"), and
    /// the two keys end up holding the one row `rules/README.md` §3.2 asks
    /// for. Without this the row would go to `SCALAR_OP`, whose emitter for a
    /// data-origin value is the row of the operator the AST names, and an
    /// operator whose value is several term nodes has no such row.
    ///
    /// Phase 0 gives a data-origin scalar a structured term only where the
    /// operator's whole arithmetic is in the term language:
    /// `np.unravel_index` and `np.ravel_multi_index` over a constant shape
    /// (`../phase0/definition.md` §5.4).
    fn structured_term_delegation(
        &self,
        request: &RowRequest,
        key: &ValueKey,
    ) -> Option<RowRequest> {
        if request.family != Family::Wire {
            return None;
        }
        let entry = self.entry(key)?;
        let Producer::Applied { rules, .. } = &entry.producer else {
            return None;
        };
        if !rules
            .iter()
            .any(|rule| matches!(rule, crate::phase0::rules::Rule::EArithData))
        {
            return None;
        }
        let scalar = entry.desc.as_scalar()?;
        let (target, index) = match scalar.term {
            ScalarTerm::Int(t) => (Target::Term(t), t.index() as u32),
            ScalarTerm::Bool(b) => (Target::BoolTerm(b), b.index() as u32),
            ScalarTerm::Opaque => return None,
        };
        if matches!(
            self.terms.node_at(index as usize),
            NodeView::Const(_) | NodeView::Var(_) | NodeView::BVar(_)
                | NodeView::True | NodeView::False
        ) {
            return None;
        }
        self.redirect(request, target, Family::Wire)
    }

    fn term_delegation(&self, request: &RowRequest, index: u32) -> Option<RowRequest> {
        if request.family != Family::Wire {
            return None;
        }
        let owner = self.term_owner.get(&index)?;
        self.redirect(request, Target::Value(owner.clone()), Family::Wire)
    }

    /// The same request aimed at another row, unless that row is on the
    /// stack of rows being filled.
    fn redirect(
        &self,
        request: &RowRequest,
        target: Target,
        family: Family,
    ) -> Option<RowRequest> {
        if self.table.is_filling_target(&target, family) {
            return None;
        }
        Some(RowRequest {
            epoch: self.epoch_for(&target, &request.filter),
            target,
            family,
            ..request.clone()
        })
    }

    /// Run every rule of the library whose output is in the row's family,
    /// build the nodes, prune, and return the row.
    ///
    /// The rules run first and the nodes are attached after, because whether
    /// this fill is provisional is only known once every rule has asked for
    /// its ports, and a provisional fill's requirement nodes carry its fill
    /// number (§4, "Acyclicity by construction").
    fn fill(&mut self, request: &RowRequest) -> Result<Row, Reject> {
        let mut row = Row::default();
        let mut offered: Vec<Candidate> = Vec::new();
        let mut tried: Vec<&'static str> = Vec::new();
        let before = self.oracle.answers.len() as u32;
        for rule in RULES {
            if !rule.answers().contains(&request.family) {
                continue;
            }
            tried.push(rule.id().as_str());
            let cut_before = self.provisional.last().copied().flatten().is_some();
            let candidates = rule.build(self, request)?;
            let cut_here = !cut_before && self.provisional.last().copied().flatten().is_some();
            if cut_here && candidates.is_empty() {
                // Which provider closes the cycle: this one.
                row.notes.push(format!(
                    "{} was dropped in the row of {}: one of its ports asked for \
                     a row that is being filled already",
                    rule.id().as_str(),
                    request.target.render()
                ));
            }
            offered.extend(candidates);
        }
        let provisional = self.provisional.last().copied().flatten();
        let mut built: Vec<ImplId> = Vec::new();
        for candidate in offered {
            if !request.filter.admits_epoch(candidate.epoch) {
                continue;
            }
            if self.graph.implementations.len() >= self.omega.node_budget {
                return Err(Reject::NodeBudget { budget: self.omega.node_budget });
            }
            if let Some(at) = self.attach(request, candidate, provisional, &mut row) {
                built.push(at);
            }
        }
        self.prune(&built);
        self.attribute_answers(before, &built);
        if row.is_empty() && self.relabels(&request.target) {
            // The row is an alias's, and the row it would have delegated to
            // is still being filled, so this one is inside a cycle. It has
            // no providers of its own and nothing to report.
            self.mark_provisional();
            return Ok(row);
        }
        if row.is_empty() {
            let note = format!(
                "no rule provides {} for {}{}; the library offers {} for that family",
                request.family.as_str(),
                request.target.render(),
                self.provenance(&request.target),
                if tried.is_empty() {
                    "none".to_string()
                } else {
                    tried.join(", ")
                }
            );
            self.missing.push(note.clone());
            row.notes.push(note);
        }
        Ok(row)
    }

    /// Record the questions this row asked that no node of it names.
    ///
    /// §2 asks that A hold exactly the questions the nodes asked. A rule
    /// that matched, asked its premises and then found no provider for a
    /// port asked them all the same, so the question is in A and belongs to
    /// no node; A marks it as the construction's rather than a node's, and
    /// the invariant checker reads that mark.
    fn attribute_answers(&mut self, before: u32, built: &[ImplId]) {
        let after = self.oracle.answers.len() as u32;
        if after == before {
            return;
        }
        let attached: std::collections::HashSet<AnswerId> = built
            .iter()
            .flat_map(|&at| self.graph.implementation(at).answers.iter().copied())
            .collect();
        for id in (before..after).map(AnswerId) {
            if !attached.contains(&id) {
                self.oracle.answers.mark_root(id);
            }
        }
    }

    /// Whether the target is a value that only relabels another.
    fn relabels(&self, target: &Target) -> bool {
        match target {
            Target::Value(key) => self
                .entry(key)
                .map(|entry| entry.producer.relabels().is_some())
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Which rule of the library would provide a row, for the note an empty
    /// one carries: for a value, the transfer rules phase 0 recorded for it
    /// (`rules/README.md` §9); for a variable of Φ, the rule that answers
    /// the value that introduced it (§2).
    fn provenance(&self, target: &Target) -> String {
        let key = match target {
            Target::Value(key) => key,
            Target::Term(t) => return self.variable_provenance(t.index() as u32),
            Target::BoolTerm(b) => return self.variable_provenance(b.index() as u32),
        };
        let Some(Producer::Applied { rules, .. }) = self.producer_of(key) else {
            return String::new();
        };
        let mut covering: Vec<&'static str> = rules
            .iter()
            .flat_map(|&rule| crate::phase1::rules::covering_rules(rule))
            .map(|rule| rule.as_str())
            .collect();
        covering.sort();
        covering.dedup();
        let names: Vec<&'static str> = rules.iter().map(|rule| rule.name()).collect();
        format!(
            ", which phase 0 produced by {} and the library covers with {}",
            names.join(", "),
            if covering.is_empty() {
                "no rule".to_string()
            } else {
                covering.join(" / ")
            }
        )
    }

    /// The rule that answers a variable of Φ, when the term is one.
    fn variable_provenance(&self, index: u32) -> String {
        let Some(term) = self.terms.int_at(index) else {
            return String::new();
        };
        let Some(variable) = self.terms.var_of(term) else {
            return String::new();
        };
        format!(
            ", the variable {} of Φ, which {} answers",
            self.terms.var_name(variable),
            rule_for_variable(self.var_origin(variable)).as_str()
        )
    }

    fn producer_of(&self, key: &ValueKey) -> Option<&Producer> {
        self.entry(key).map(|entry| &entry.producer)
    }

    /// Create the candidate's requirement nodes and attach the
    /// implementation node to them.
    fn attach(
        &mut self,
        request: &RowRequest,
        candidate: Candidate,
        provisional: Option<u32>,
        row: &mut Row,
    ) -> Option<ImplId> {
        let mut outputs = Vec::with_capacity(candidate.outputs.len());
        let mut pending = Vec::with_capacity(candidate.outputs.len());
        for output in &candidate.outputs {
            // §2: a bundle that answers a `Fetch` names the value and the
            // batch; every other bundle names the target and the interface.
            let form = match (&output.batch, output.target.as_value()) {
                (Some(batch), Some(value)) => Form::Fetch {
                    value: value.clone(),
                    batch: batch.clone(),
                },
                _ => Form::Represent {
                    target: output.target.clone(),
                    interface: output.interface.clone(),
                },
            };
            let id = RequirementId {
                form,
                condition: output.condition,
                sharing: request.sharing,
                filter: request.filter.clone(),
                epoch: Some(candidate.epoch),
                provisional,
            };
            // A requirement that already exists is one another row created,
            // and attaching to it may close a cycle; a fresh one cannot.
            let existed = self.graph.find(&id).is_some();
            let at = self.graph.requirement(id);
            // The bundle belongs to this row when it names the row's target
            // and family, and — for a bundle that answers a `Fetch` — the
            // row's batch. A `Represent` bundle carries no batch, and the
            // rows that are keyed by one (`Coords`, `ReadReq`) hold it.
            let answers_row = output.target == request.target
                && output.interface.family() == request.family
                && match &output.batch {
                    Some(batch) => request.batch.as_ref() == Some(batch),
                    None => true,
                };
            pending.push((at, existed, answers_row, output.interface.clone()));
        }
        // §3.4 lets one node answer several requirements. An attachment that
        // makes the node reachable from itself would put a cycle in G, which
        // §2 forbids, so it is not made: the node keeps the requirements that
        // do not close one, and a node that would close one on the row's own
        // requirement is not built at all.
        let closing = self.closing_attachments(&candidate.inputs, &pending);
        for &(at, _, answers_row, ref interface) in &pending {
            if closing.contains(&at) {
                if answers_row {
                    row.notes.push(format!(
                        "{} answers {} and reads it, so attaching it would close \
                         a cycle in G",
                        candidate.rule.as_str(),
                        request.target.render()
                    ));
                    return None;
                }
                continue;
            }
            outputs.push(at);
            if answers_row && !row.entries.iter().any(|entry| entry.requirement == at) {
                row.entries.push(RowEntry {
                    interface: interface.clone(),
                    requirement: at,
                    epoch: candidate.epoch,
                });
            }
        }
        if outputs.is_empty() {
            return None;
        }
        let cost = self.cost.emitter(&candidate.primitives);
        Some(self.graph.add_implementation(ImplNode {
            rule: candidate.rule,
            matched: candidate.matched,
            inputs: candidate.inputs,
            outputs,
            parameters: candidate.parameters,
            epoch: candidate.epoch,
            cost,
            primitives: candidate.primitives,
            answers: candidate.answers,
            readers: candidate.readers,
            variant: candidate.variant,
            dropped: false,
        }))
    }

    /// Which of a candidate's output requirements are reachable from its
    /// inputs, and so would close a cycle if the node were attached to them.
    ///
    /// Only a requirement that existed before this node can be: a fresh one
    /// has no provider yet and nothing reaches it. The search therefore runs
    /// only where §3.4 attaches to a requirement another row created.
    fn closing_attachments(
        &self,
        inputs: &[crate::phase1::graph::ReqId],
        pending: &[(crate::phase1::graph::ReqId, bool, bool, crate::phase1::interface::Interface)],
    ) -> BTreeSet<crate::phase1::graph::ReqId> {
        let targets: BTreeSet<crate::phase1::graph::ReqId> = pending
            .iter()
            .filter(|&&(_, existed, _, _)| existed)
            .map(|&(at, _, _, _)| at)
            .collect();
        let mut found = BTreeSet::new();
        if targets.is_empty() || inputs.is_empty() {
            return found;
        }
        let mut seen: BTreeSet<crate::phase1::graph::ReqId> = BTreeSet::new();
        let mut stack: Vec<crate::phase1::graph::ReqId> = inputs.to_vec();
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            if targets.contains(&at) {
                found.insert(at);
            }
            for &provider in &self.graph.node(at).providers {
                stack.extend(self.graph.implementation(provider).inputs.iter().copied());
            }
        }
        found
    }

    /// The dominance pruning of §4, applied to the nodes one row built.
    fn prune(&mut self, built: &[ImplId]) {
        for node in dominated(&self.graph, built) {
            self.graph.detach(node);
            self.oracle.stats.pruned += 1;
        }
    }

    // ── Roots and the order of §4 ────────────────────────────────────

    /// §4 step 1: CANFAIL, in check order.
    ///
    /// For check k, ask whether Φ_{position(k)−1} ∧ φ_k ∧ ¬defcond_k has a
    /// model. "Yes" and "unknown" make k an `Enforce(k)` root; "no" means k
    /// is implied by the ranges, the declarations and the checks before it.
    pub fn canfail_sweep(&mut self) -> Vec<EnforceTarget> {
        let mut roots = Vec::new();
        // The checks, through the oracle's incremental sweep (`oracle::sweep`);
        // it returns them in constraint order.
        let mut checks = self.oracle.canfail_sweep(&self.terms).into_iter().peekable();
        let constraints = self.result.phi.constraints().len();
        for at in 0..constraints {
            let constraint = &self.result.phi.constraints()[at];
            match constraint.kind {
                ConstraintKind::Check { .. } => {
                    let (index, answer) = checks.next().expect("one answer per check");
                    debug_assert_eq!(index, at);
                    if self.verdict_of(answer) != Verdict::CanFail(false) {
                        roots.push(EnforceTarget::Check { constraint: at });
                    }
                }
                // `../README.md` §2: a `@requires` relating several
                // parameters is a check the circuit enforces on the inputs.
                ConstraintKind::Precondition => {
                    let formula = constraint.formula;
                    let variables = self
                        .oracle
                        .layers
                        .variables_of(&self.terms, formula.index() as u32)
                        .len();
                    if variables > 1 {
                        roots.push(EnforceTarget::Precondition { constraint: at });
                    }
                }
                _ => {}
            }
        }
        roots
    }

    /// §4 step 2 and 3: the roots and the depth-first expansion.
    pub fn expand(&mut self, enforce: &[EnforceTarget]) -> Result<(), Reject> {
        let delivery: Vec<ValueKey> = self.result.delivery.clone();
        for key in delivery {
            self.export_root(&key)?;
        }
        for &target in enforce {
            self.enforce_root(target)?;
        }
        Ok(())
    }

    /// One `Represent(v, Export)` per key of R (§2, "Roots"), and one per
    /// element of a composite in R, in element order (`export.md`).
    fn export_root(&mut self, key: &ValueKey) -> Result<(), Reject> {
        let Some(entry) = self.entry(key) else {
            return Err(Reject::NoDescriptor { value: key.encode() });
        };
        match entry.desc.clone() {
            Desc::Composite(composite) => {
                let epoch = self.watermark(key);
                for at in 0..composite.elems.len() {
                    let path = [at];
                    let target = self.delivered_element(key, &path)?;
                    let condition = self.element_psi(key, &path);
                    self.root_of(target, condition, epoch)?;
                }
                Ok(())
            }
            // A value with no descriptor content delivers nothing.
            Desc::Class(_) | Desc::None => Ok(()),
            _ => {
                let condition = self.psi(key);
                let epoch = self.watermark(key);
                self.root_of(Target::Value(key.clone()), condition, epoch)
            }
        }
    }

    /// The target of one element of a delivered composite, or the rejection
    /// that names why the element has none.
    ///
    /// [`Builder::element_target`] is the shared reading of "the element of a
    /// composite by its path" (`rules/README.md` §2); this wraps it in the
    /// message the delivery needs, so that an element with no name is
    /// refused rather than silently dropped.
    fn delivered_element(&self, key: &ValueKey, path: &[usize]) -> Result<Target, Reject> {
        if let Some(target) = self.element_target(key, path) {
            return Ok(target);
        }
        let requirement = format!("Represent({}, Export)", key.encode());
        let detail = match self.element_of(key, path) {
            Some(Desc::Scalar(scalar)) => format!(
                "an element of the delivered composite is a {} scalar, which \
                 carries no term and so has no wire the library states",
                scalar.dtype.as_str()
            ),
            Some(other) => format!(
                "an element of the delivered composite is a {}, which Δ keys \
                 only as part of the composite",
                other.kind_str()
            ),
            None => "the delivered composite has no element there".to_string(),
        };
        Err(Reject::NoProvider { requirement, detail })
    }

    fn root_of(&mut self, target: Target, condition: BoolId, epoch: u32) -> Result<(), Reject> {
        let request = RowRequest {
            target: target.clone(),
            family: Family::Export,
            condition,
            sharing: SharingId::PRIMARY,
            filter: EpochFilter::default(),
            epoch,
            batch: None,
            axes: None,
            extent: None,
            ranks: None,
        };
        // The notes this root's own fill added, so the rejection names the
        // first empty row *under* the failing root rather than the first of
        // the whole construction, which an earlier root may have left.
        let before = self.missing.len();
        let row = self.row(&request)?;
        if row.is_empty() {
            return Err(Reject::NoProvider {
                requirement: format!("Represent({}, Export)", target.render()),
                detail: self
                    .missing
                    .get(before)
                    .or_else(|| self.missing.first())
                    .cloned()
                    .unwrap_or_else(|| row.notes.join("; ")),
            });
        }
        for entry in &row.entries {
            if entry.interface.family() == Family::Export {
                self.graph.roots.push(entry.requirement);
            }
        }
        Ok(())
    }

    /// One `Enforce(k)` per check the sweep kept, and per multi-variable
    /// `precondition` (§2, "Roots"; §4 step 5).
    fn enforce_root(&mut self, target: EnforceTarget) -> Result<(), Reject> {
        let position = match target {
            EnforceTarget::Check { constraint } => self.oracle.layers.position(constraint),
            EnforceTarget::Precondition { .. } => None,
        };
        let filter = match position {
            Some(position) => EpochFilter::default().below_check(position),
            // A `precondition` holds by declaration in every Φ_e, so it has
            // no position in the check order and no filter of its own.
            None => EpochFilter::default(),
        };
        // The root is raised under the subject's own execution condition,
        // not under `true`: φ_k is the path condition at the operation, so
        // every value φ_k and defcond_k mention was computed on that path,
        // and a port under `true` would ask for a wire the program has only
        // where ψ holds (§2, "Roots"). `ENFORCE_CHECK` gates the constraint
        // by ψ's own wire, which is what keeps the check vacuous off that
        // path.
        let subject = self.enforce_subject(target);
        let condition = match target {
            // A `@requires` fact holds before the body runs, so its root is
            // raised under true; `enforce_subject` has no key for it.
            EnforceTarget::Precondition { .. } => self.terms.tt(),
            EnforceTarget::Check { .. } => self.psi(&subject),
        };
        let request = RowRequest {
            target: Target::Value(subject),
            family: Family::Wire,
            condition,
            sharing: SharingId::PRIMARY,
            filter: filter.clone(),
            epoch: position.map(|p| p.saturating_sub(1)).unwrap_or(0),
            batch: None,
            axes: None,
            extent: None,
            ranks: None,
        };
        let id = RequirementId {
            form: Form::Enforce { target },
            condition,
            sharing: SharingId::PRIMARY,
            filter,
            epoch: None,
            provisional: None,
        };
        let at = self.graph.requirement(id);
        self.graph.roots.push(at);
        let candidates = crate::phase1::rules::scalar_control::enforce_check::build_root(
            self, target, &request,
        )?;
        if candidates.is_empty() {
            return Err(Reject::NoProvider {
                requirement: format!("Enforce({})", target.constraint()),
                detail: self
                    .missing
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "ENFORCE_CHECK found no wire for the check's terms".to_string()),
            });
        }
        for candidate in candidates {
            let cost = self.cost.emitter(&candidate.primitives);
            self.graph.add_implementation(ImplNode {
                rule: candidate.rule,
                matched: candidate.matched,
                inputs: candidate.inputs,
                outputs: vec![at],
                parameters: candidate.parameters,
                epoch: candidate.epoch,
                cost,
                primitives: candidate.primitives,
                answers: candidate.answers,
                readers: candidate.readers,
                variant: candidate.variant,
                dropped: false,
            });
        }
        Ok(())
    }

    /// The value a check belongs to, which is the key phase 0 recorded as
    /// its subject.
    fn enforce_subject(&self, target: EnforceTarget) -> ValueKey {
        match self.result.phi.constraints()[target.constraint()].kind {
            ConstraintKind::Check { ref subject, .. } => subject.clone(),
            _ => ValueKey {
                stmt_path: Vec::new(),
                iters: Vec::new(),
                calls: Vec::new(),
                expr_index: u32::MAX,
                step: 0,
            },
        }
    }

    /// The value a key names after the relabellings of `rules/README.md` §2:
    /// the *defining key* the count of a mask is keyed by
    /// (`masks/README.md` §2), an `Alias`, an `E-Copy` and a single-operand
    /// `S-Assign` followed.
    pub fn defining_key(&self, key: &ValueKey) -> ValueKey {
        let mut current = key.clone();
        for _ in 0..self.result.descriptors.len() {
            let Some(entry) = self.entry(&current) else {
                break;
            };
            let Some(source) = entry.producer.relabels() else {
                break;
            };
            if self.entry(source).map(|other| other.desc.kind_str())
                != Some(entry.desc.kind_str())
            {
                break;
            }
            current = source.clone();
        }
        current
    }

    /// `c = Count(m)` as a term, for the mask a rule matched
    /// (`masks/README.md` §2): the variable of Φ whose origin names this
    /// mask's defining key, `None` when the mask was never used as a
    /// selector and Φ holds no count for it.
    pub fn count_of(&mut self, mask: &ValueKey) -> Option<TermId> {
        let defining = self.defining_key(mask);
        let found = self.result.phi.vars().iter().enumerate().find(|(_, info)| {
            matches!(&info.origin, VarOrigin::Count { mask } if mask == &defining)
        })?;
        let variable = crate::phase0::term::VarId(found.0 as u32);
        Some(self.terms.var(variable))
    }

    /// The row a term's wire lives in (`rules/README.md` §3.2): the value of
    /// Δ that computes the term when there is one, and the term itself
    /// otherwise.
    ///
    /// A node that outputs a wire for a term — `COUNT`'s `r_B` — has to name
    /// the same target wherever it is offered, or the row of the term and
    /// the row of the value that carries it would hold two nodes instead of
    /// the one node §3.4 asks for. This is the target
    /// [`Builder::delegation`] would send the term row to.
    pub fn wire_target(&self, term: TermId) -> Target {
        match self.owner_of_term(term.index() as u32) {
            Some(owner) => Target::Value(owner.clone()),
            None => Target::Term(term),
        }
    }

    /// The first array value of Δ one of whose axes is this term.
    ///
    /// `rules/README.md` §2 resolves ℓ and c′ to "the value they size". For
    /// `unique` with `return_inverse` the variable's origin names the Tuple
    /// the call produced, which Δ keys as a Composite, so the value that
    /// carries the extent is the element bound out of it; this finds it by
    /// the extent itself.
    pub fn array_with_dim(&self, term: TermId) -> Option<ValueKey> {
        self.result
            .descriptors
            .iter()
            .find(|entry| {
                entry
                    .desc
                    .as_array()
                    .map(|array| array.dims.contains(&term))
                    .unwrap_or(false)
            })
            .map(|entry| entry.key.clone())
    }

    /// The row a `Ranks` port asks for: the rank chain of one bundle of one
    /// value, under that value's own ψ (`rules/README.md` §4.5).
    pub fn ranks_port(
        &mut self,
        candidate: &mut Candidate,
        request: &RowRequest,
        value: &ValueKey,
        subject: RanksOf,
    ) -> Result<Row, Reject> {
        let psi = self.psi(value);
        let condition = self.child_under_g(candidate, request, psi, &value.encode())?;
        let target = Target::Value(value.clone());
        let epoch = self.epoch_for(&target, &request.filter);
        let keyed = request.child_ranks(target, subject, condition, epoch);
        let row = self.row(&keyed)?;
        if row.cut {
            self.mark_provisional();
        }
        Ok(row)
    }

    /// The `[a, b]` phase 0 recorded for a variable, as integers.
    pub fn var_range(&self, v: VarId) -> (Option<i128>, Option<i128>) {
        let info = self.result.phi.var_info(v);
        (
            info.lo.and_then(|bound| self.interval(bound).lo),
            info.hi.and_then(|bound| self.interval(bound).hi),
        )
    }

    pub fn var_origin(&self, v: VarId) -> &VarOrigin {
        &self.result.phi.var_info(v).origin
    }
}

/// The conjunction of `parts`, in canonical form.
///
/// `revision/phase1/README.md` §4, "When a question is asked": the condition
/// of a question is a Boolean term phase 1 builds by conjunction. Two
/// spellings of one conjunct set must be one term id, or the memo of A
/// misses and the same question is asked twice, so the conjuncts are
/// flattened, deduplicated and folded in arena-id order. `true` when the
/// parts have no conjunct between them; `false` as soon as one is.
pub fn conjunction(terms: &mut TermArena, parts: &[BoolId]) -> BoolId {
    let mut conjuncts: Vec<u32> = Vec::new();
    for &part in parts {
        conjuncts.extend(layers::conjuncts(terms, part));
    }
    conjuncts.sort_unstable();
    conjuncts.dedup();
    let mut out = terms.tt();
    for at in conjuncts {
        // Every conjunct came out of `conjuncts`, so it is Boolean-sorted.
        if let Some(formula) = terms.bool_at(at) {
            out = terms.and(out, formula);
        }
    }
    out
}

/// One capacity box of a value, at one epoch (`rules/README.md` §6.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityBox {
    /// The epoch the MAX answers were taken at, which is the node's.
    pub epoch: u32,
    /// `B_k` per axis.
    pub axes: Vec<Capacity>,
    /// Whether axis `k` was decided by CONST, so that no shape wire is
    /// needed for it.
    pub constant: Vec<bool>,
}

impl CapacityBox {
    /// `Q = ∏_a B_a`, the slots of the box.
    pub fn slots(&self) -> Capacity {
        self.axes.iter().product()
    }
}

/// The nodes of a row the dominance pruning of §4 drops.
///
/// "Drop an implementation node whose outputs and child requirements both
/// equal another's in the row and whose cost is not lower. Nodes with
/// different children are never compared: their children's costs are phase
/// 2's to weigh." This is the only cost comparison the construction makes.
pub(crate) fn dominated(graph: &Graph, built: &[ImplId]) -> Vec<ImplId> {
    let mut dropped: Vec<ImplId> = Vec::new();
    for (at, &node) in built.iter().enumerate() {
        for &other in built.iter().take(at) {
            if node == other || dropped.contains(&other) || dropped.contains(&node) {
                continue;
            }
            let (a, b) = (graph.implementation(node), graph.implementation(other));
            if a.outputs != b.outputs || a.inputs != b.inputs {
                continue;
            }
            // The later node is dropped on a tie, so the result does not
            // depend on the order the rules ran in.
            if a.cost >= b.cost {
                dropped.push(node);
            } else {
                dropped.push(other);
            }
        }
    }
    dropped
}

/// `W(t) = ⌈log2 max(1, −lo, hi + 1)⌉` of `rules/README.md` §6.1.
pub fn width_of(range: Interval) -> u32 {
    let magnitude = [
        Some(1i128),
        range.lo.map(|lo| -lo),
        range.hi.map(|hi| hi + 1),
    ]
    .into_iter()
    .flatten()
    .max()
    .unwrap_or(1)
    .max(1);
    bits_for(magnitude)
}

/// `⌈log2 n⌉` for `n ≥ 1`.
pub fn bits_for(n: i128) -> u32 {
    let mut width = 0;
    let mut covered: i128 = 1;
    while covered < n {
        covered = covered.saturating_mul(2);
        width += 1;
    }
    width
}

/// `bits(B) = ⌈log2(B + 1)⌉` of §6.1: the width of a wire over a capacity.
pub fn bits_of_capacity(capacity: i128) -> u32 {
    bits_for(capacity.max(0) + 1)
}

/// The rule that would answer a variable of Φ, for the "no provider in step
/// 1" note of `rules/README.md` §2.
pub fn rule_for_variable(origin: &VarOrigin) -> RuleId {
    match origin {
        VarOrigin::Count { .. } => RuleId::Count,
        VarOrigin::InputLength { .. } | VarOrigin::ShapeSymbol { .. } => RuleId::LenShape,
        VarOrigin::ValueSized { .. } | VarOrigin::UniqueCount { .. } => RuleId::LenShape,
        VarOrigin::SizeProduct { .. } => RuleId::ShapeCheck,
        VarOrigin::DataScalar { .. } | VarOrigin::DataBool { .. } => RuleId::ScalarOp,
        VarOrigin::Definedness { .. } => RuleId::TermToWires,
    }
}
