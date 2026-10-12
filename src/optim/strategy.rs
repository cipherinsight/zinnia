//! Op strategy selection by property queries (Phase F foundation).
//!
//! An op with multiple correct lowerings declares them as an ordered
//! [`OpStrategySet`]: a list of [`OpStrategy`] entries, each gated by a
//! [`ContractTerm`] precondition, plus a `default` fall-through lowering.
//! The [`dispatch_strategy`] helper walks the list, calls [`IRBuilder::prove`]
//! on each precondition, and runs the first one whose outcome is
//! [`ProveOutcome::Proved`]. If no gated strategy is proved, the `default`
//! runs.
//!
//! ## Soundness invariant
//!
//! Every strategy's `lower` function MUST be sound under its `precondition`,
//! and `default` MUST be sound unconditionally. The dispatcher only chooses
//! among lowerings the op author has already certified as sound; it never
//! relaxes a check or downgrades [`ProveOutcome::Unknown`] to "good enough".
//! [`ProveOutcome::Unknown`] and [`ProveOutcome::Disproved`] both cause the
//! gated strategy to be skipped — the dispatcher falls through to the next
//! candidate, ultimately landing on `default`.
//!
//! ## Multiple matches
//!
//! When two strategies' preconditions are both currently `Proved`, the
//! **first declared** wins: the op author orders strategies by cost.
//!
//! ## Precondition construction
//!
//! The `precondition` is a plain [`ContractTerm`] built by the op author at
//! the call site (e.g., `is_sorted(arr_vid)`). Unlike contract templates,
//! strategies do not use `Var(Formal(_))` substitution because each strategy
//! set is per-call: the op author already has the concrete `ValueId`s in
//! scope and embeds them directly as `Var(Value(_))` leaves.

use crate::builder::IRBuilder;
use crate::optim::predicates::formula::ContractTerm;
use crate::optim::prove::ProveOutcome;

/// One alternative lowering for an op, gated by a Boolean property of the
/// op's inputs.
///
/// ## Soundness contract
///
/// `lower` MUST produce IR that computes the op's specified semantics
/// **whenever `precondition` holds**. The dispatcher refuses to call `lower`
/// unless [`IRBuilder::prove`] returns [`ProveOutcome::Proved`] for
/// `precondition`, so a strategy whose lowering exploits the property is
/// only ever reached when that property is provable from the visible facts.
///
/// `name` is used for telemetry only; it is not load-bearing.
pub struct OpStrategy<Inputs, Output> {
    pub name: &'static str,
    pub precondition: ContractTerm,
    pub lower: fn(&mut IRBuilder, &Inputs) -> Output,
}

/// The precondition of a strategy that is offered only when its property
/// holds. Properties of an array's elements (all zero, all one, an identity
/// matrix, sorted, in a range) are array facts (`builder/array_facts.rs`),
/// which describe the array's current contents and never reach `prove`: the
/// strategy set asks them when it is built, so the dispatcher has nothing
/// left to prove.
pub fn decided() -> ContractTerm {
    ContractTerm::LitBool(true)
}

/// An ordered set of alternative lowerings for one op invocation plus a
/// `default` that runs when no gated strategy fires.
///
/// ## Soundness contract
///
/// `default` MUST be sound **unconditionally** — it is the fall-through and
/// runs whenever every gated strategy's precondition is either `Unknown` or
/// `Disproved`. Every entry in `strategies` MUST be sound under its own
/// precondition (see [`OpStrategy`]).
pub struct OpStrategySet<Inputs, Output> {
    pub strategies: Vec<OpStrategy<Inputs, Output>>,
    pub default: fn(&mut IRBuilder, &Inputs) -> Output,
}

/// Walk `set.strategies` in declared order and run the first whose
/// `precondition` proves; fall through to `set.default` if none fire.
///
/// ## Soundness contract
///
/// The dispatcher's contract is: it only ever invokes a lowering the op
/// author has already certified as sound. Specifically:
///
/// - It calls a gated strategy's `lower` **only** when
///   [`IRBuilder::prove`] returns [`ProveOutcome::Proved`] for that
///   strategy's `precondition`. [`ProveOutcome::Unknown`] and
///   [`ProveOutcome::Disproved`] both skip the strategy — `Unknown` is
///   never silently treated as `Proved`.
/// - It calls `set.default` only when no gated strategy fired, and `default`
///   must be sound unconditionally.
///
/// Together these ensure dispatch_strategy can only change *which sound
/// lowering* runs, never admit an unsound one.
///
/// ## Telemetry
///
/// Emits a structured `tracing::debug!` line per non-default fire:
/// `op_strategy_dispatch: op=<op> strategy=<strat>`, on target
/// `zinnia::op_strategy`. The `op` label is supplied by the caller via
/// `op_name`; default-fall-through is not logged to keep current builds
/// quiet (matches the precedent set by Phase E discharge telemetry).
pub fn dispatch_strategy<Inputs, Output>(
    b: &mut IRBuilder,
    op_name: &'static str,
    inputs: &Inputs,
    set: &OpStrategySet<Inputs, Output>,
) -> Output {
    // A/B-harness kill switch: under `ZINNIA_REQ_DISABLE=1` skip the
    // gated strategies entirely and always run the default lowering.
    // Default is sound unconditionally (see `OpStrategySet` doc), so
    // this is safe; it just stops specialisation.
    if crate::optim::resolver::req_disabled() {
        if let Some(sink) = &b.telemetry {
            sink.emit(&crate::optim::telemetry::TelemetryEvent::StrategyDefault {
                op: op_name.to_string(),
            });
        }
        return (set.default)(b, inputs);
    }
    for strat in &set.strategies {
        if matches!(b.prove(&strat.precondition), ProveOutcome::Proved) {
            tracing::debug!(
                target: "zinnia::op_strategy",
                "op_strategy_dispatch: op={} strategy={}",
                op_name,
                strat.name,
            );
            // Structured telemetry mirror of the tracing line. The
            // precondition term's ValueId leaves are summarized as
            // input_value_ids so an A/B harness can attribute dispatches
            // back to specific SSA inputs.
            if let Some(sink) = &b.telemetry {
                let value_ids = crate::optim::predicates::collect_value_ids(&strat.precondition);
                sink.emit(&crate::optim::telemetry::strategy_dispatch_event(
                    op_name, strat.name, &value_ids,
                ));
            }
            return (strat.lower)(b, inputs);
        }
    }
    // No gated strategy fired — record the fall-through so an A/B harness
    // can directly count machinery-engagement-rate.
    if let Some(sink) = &b.telemetry {
        sink.emit(&crate::optim::telemetry::TelemetryEvent::StrategyDefault {
            op: op_name.to_string(),
        });
    }
    (set.default)(b, inputs)
}
