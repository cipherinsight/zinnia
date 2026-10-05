//! `ENFORCE_CHECK` (`revision/phase1/rules/scalar_control/enforce_check.md`).
//!
//! Turns a check of Φ into one constraint. A check is `φ_k ⇒ defcond_k`;
//! phase 1 makes `Enforce(k)` a root for every k whose CANFAIL answer is
//! "can fail" or "unknown" (`../../README.md` §4 step 1), and this rule
//! answers it by obtaining the wires of `φ_k` and `defcond_k` as term
//! requirements and constraining `φ_k ⇒ defcond_k`, so that a run which
//! fails at the operation has no witness.
//!
//! Every row filled below `Enforce(k)` carries the epoch filter of
//! `../../README.md` §4 step 5, epoch < position(k): every wire the
//! constraint reads is computed by nodes that assumed no check at or after
//! k. The filter lives in the root's request, so
//! nothing here propagates it.

use crate::phase0::phi::ConstraintKind;

use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::graph::{EnforceTarget, Target};
use crate::phase1::interface::Family;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct EnforceCheck;

impl Rule for EnforceCheck {
    fn id(&self) -> RuleId {
        RuleId::EnforceCheck
    }

    /// `Enforce(k)` is a root, not a family a consumer asks a row for, so
    /// the rule matches no row; [`build_root`] is how the construction
    /// reaches it.
    fn answers(&self) -> &'static [Family] {
        &[]
    }

    fn build(&self, _: &mut Builder<'_>, _: &RowRequest) -> Result<Vec<Candidate>, Reject> {
        Ok(Vec::new())
    }
}

/// The implementation nodes answering one `Enforce` root.
pub(crate) fn build_root(
    builder: &mut Builder<'_>,
    target: EnforceTarget,
    request: &RowRequest,
) -> Result<Vec<Candidate>, Reject> {
    let constraint = &builder.result.phi.constraints()[target.constraint()];
    // source_pattern: the check record of Φ. A `precondition` root has
    // φ = true and the constraint itself as the condition (`../README.md` §2).
    let (path, defcond) = match constraint.kind {
        ConstraintKind::Check { path, defcond, .. } => (path, defcond),
        _ => (builder.terms.tt(), constraint.formula),
    };
    let mut candidate = Candidate::new(RuleId::EnforceCheck, None, request.epoch)
        .parameter("position", target.constraint() as i128);

    // `gate`: one port per conjunct of ψ, the subject's execution condition,
    // which the root carries as its own condition (`../../build.rs`,
    // `enforce_root`). ψ is a path condition b_1 ∧ … ∧ b_C, and b_i was
    // computed under b_1 ∧ … ∧ b_{i−1}, so the conjuncts are raised in that
    // nesting and not all under true; they are ordered by term id, which is
    // the order phase 0 built them in and therefore the nesting order. The
    // whole group is absent where ψ is the constant true and where CONST
    // decides it true, which is every check outside a branch.
    let gate = builder.under(request.condition);
    let mut gated = 0usize;
    if !decided_true(builder, &mut candidate, request, gate) {
        let mut so_far = builder.terms.tt();
        for conjunct in gate_conjuncts(builder, gate) {
            if !port(builder, request, &mut candidate, conjunct, so_far, "ψ")? {
                return Ok(Vec::new());
            }
            so_far = crate::phase1::build::conjunction(
                &mut builder.terms,
                &[so_far, conjunct],
            );
            gated += 1;
        }
    }
    // `path`: `Represent(φ_k, Wire[Bool])` under ψ; absent when φ_k is
    // the constant true.
    if !builder.terms.is_true(path) {
        if !port(builder, request, &mut candidate, path, gate, "φ")? {
            return Ok(Vec::new());
        }
    }
    // `cond`: `Represent(defcond_k, Wire[Bool])` under ψ ∧ φ_k; absent when
    // defcond_k is the constant false, which is the `while` cap form.
    if !builder.terms.is_false(defcond) {
        let under = crate::phase1::build::conjunction(&mut builder.terms, &[gate, path]);
        if !port(builder, request, &mut candidate, defcond, under, "defcond")? {
            return Ok(Vec::new());
        }
    }

    // emitter: `g̃` is the `and` of the gate's C wires, C − 1; then
    // `mul(g̃, φ̃)` and `mul(·, 1 − d̃) = 0`, one `mul` per factor that is
    // not the constant 1. The general form is 2, the ungated one 1, and the
    // two degenerate ones a `lin`.
    if gated > 1 {
        candidate = candidate.emit(Primitive::And, gated as i128 - 1);
    }
    if !builder.terms.is_false(defcond) {
        let factors =
            usize::from(gated > 0) + usize::from(!builder.terms.is_true(path));
        candidate = candidate.emit(Primitive::Mul, factors as i128);
    }
    Ok(vec![candidate])
}

/// The conjuncts of ψ in the order phase 0 built them, which is the order a
/// path condition nests in: a branch taken inside another was recorded after
/// it, so its term has the larger id.
fn gate_conjuncts(
    builder: &Builder<'_>,
    gate: crate::phase0::term::BoolId,
) -> Vec<crate::phase0::term::BoolId> {
    crate::phase1::oracle::layers::conjuncts(&builder.terms, gate)
        .into_iter()
        .filter_map(|at| builder.terms.bool_at(at))
        .collect()
}

/// Whether the gate is the constant true, by the term or by CONST over
/// Φ_e (`enforce_check.md`, `premises` P2).
fn decided_true(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    request: &RowRequest,
    gate: crate::phase0::term::BoolId,
) -> bool {
    if builder.terms.is_true(gate) {
        return true;
    }
    let always = builder.terms.tt();
    let answer = builder.ask_for(
        candidate,
        crate::phase1::oracle::questions::Question::ConstBool(gate),
        request.epoch,
        always,
    );
    matches!(
        answer,
        crate::phase1::oracle::answers::Verdict::ConstantBool(true)
    )
}

/// One of the two ports, with the note a missing wire leaves.
///
/// A check with no wire is a rejection, and the message has to say which of
/// the two terms had none and why the row was empty; without that the report
/// names only "no wire for the check's terms" and the reader has to guess.
fn port(
    builder: &mut Builder<'_>,
    request: &RowRequest,
    candidate: &mut Candidate,
    term: crate::phase0::term::BoolId,
    under: crate::phase0::term::BoolId,
    named: &str,
) -> Result<bool, Reject> {
    let row = builder.child(request, Target::BoolTerm(term), Family::Wire, under)?;
    match row.entries.first() {
        Some(first) => {
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
            Ok(true)
        }
        None => {
            let note = format!(
                "ENFORCE_CHECK has no wire for the check's {} `{}`{}",
                named,
                builder.render_bool(term),
                if row.notes.is_empty() {
                    String::new()
                } else {
                    format!(": {}", row.notes.join("; "))
                }
            );
            if !builder.missing.contains(&note) {
                builder.missing.push(note);
            }
            Ok(false)
        }
    }
}
