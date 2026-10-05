//! The JSON form of a [`Phase1Result`], for the Python harnesses.
//!
//! # Schema
//!
//! ```text
//! {
//!   "schema":          "zinnia.phase1/5",
//!   "requirements":    [...],   // the requirement nodes of G
//!   "implementations": [...],   // the implementation nodes of G
//!   "roots":           [index, ...],
//!   "answers":         [...],   // A, in the order the questions were asked
//!   "terms":           {"base": u32, "nodes": [...]},  // the terms phase 1 built
//!   "stats":           {...},   // the oracle and graph statistics
//!   "missing":         [str],   // rows with no provider, by the rule that would fill them
//!   "violations":      [str]    // the invariant checker's findings, empty when it passes
//! }
//! ```
//!
//! `terms` carries the nodes phase 1 built over Φ's variables (`../README.md`
//! §1.1): `base` is phase 0's `node_count()`, the id at which they start, and
//! `nodes` holds them from that id upward in phase 0's own node format
//! (`crate::phase0::export`, "terms"). A reader that concatenates phase 0's
//! `terms.nodes` with these resolves every term id a requirement, a condition
//! or an answer names.
//!
//! Integers are decimal *strings* wherever they can leave the JSON-safe
//! range, as `crate::phase0::export` writes them, so that a capacity or a
//! cost survives exactly.
//!
//! A requirement is
//!
//! ```text
//! {"index": u32, "form": "represent"|"fetch"|"enforce", "target": str,
//!  "interface": str | null, "family": str | null, "condition": bool term id,
//!  "epoch": u32 | null, "filter": {"epoch_below": u32 | null},
//!  "provisional": u32 | null, "providers": [index, ...],
//!  "layout": λ | null, "batch": batch | null}
//! ```
//!
//! `provisional` is the fill number of §4's cut, `null` for a row the memo
//! kept; `sharing` is the sharing identity of `../README.md` §5.1.
//!
//! `layout` is the λ of a Store-family interface (`rules/README.md` §4.1) in
//! machine-readable form, `null` for every other family:
//!
//! ```text
//! {"kind": "box"|"compact"|"ranked"|"keyed", "slots": decimal string,
//!  "grid": [decimal string] | null,     // box: the capacity per axis
//!  "ranks": key | null,                 // ranked: the mask the ranks came from
//!  "dense": bool, "ordered": bool,
//!  "root": key, "lineage": [step, ...]}
//! ```
//!
//! and a lineage step is one of
//!
//! ```text
//! {"kind": "filter",    "mask": key}
//! {"kind": "slice",     "lo": term id, "hi": term id, "by": decimal string,
//!                       "axis": usize}
//! {"kind": "take",      "index": key, "axis": usize}
//! {"kind": "transpose", "permutation": [usize, ...]}
//! {"kind": "reshape",   "shape": [decimal string, ...]}
//! ```
//!
//! `batch` is the batch of a `Fetch` (`rules/README.md` §3.3), `null` for
//! every other form. A range batch writes its per-axis template, a general
//! batch its entry count alone:
//!
//! ```text
//! {"kind": "scalar"|"general"|"range", "entries": decimal string,
//!  "rank": usize, "live": "in_range" | bool term id,
//!  "axes": [{"base": coordinate, "stride": decimal string,
//!            "width": decimal string, "broadcast": bool term id | null},
//!           ...] | null}
//! ```
//!
//! where a coordinate is `{"kind": "literal", "value": decimal string}` or
//! `{"kind": "term", "term": term id}`.
//!
//! An implementation is
//!
//! ```text
//! {"index": u32, "rule": str, "variant": str | null, "matched": [key, ...],
//!  "inputs": [index, ...], "outputs": [index, ...],
//!  "parameters": {name: decimal string | {"lo": …, "hi": …}},
//!  "epoch": u32, "cost": decimal string,
//!  "primitives": [{"name": str, "times": decimal string, …}, ...],
//!  "fops": [{"op": backend instruction, "times": decimal string}, ...],
//!  "readers": [{"requirement": index, "queries": decimal string}, ...],
//!  "dropped": bool, "answers": [index, ...]}
//! ```
//!
//! `primitives` is the emitter's count in the primitives of
//! `rules/README.md` §7, which is what `cost` was computed from: `name` is
//! the §7 row, `times` how often it occurs, and the remaining fields are the
//! parameters that row carries (`width`, `capacity`, `arity`, `cells`, and
//! `op` for an `fop`). `fops` stays as the float instructions alone, which
//! phase 3 reads.
//!
//! and an answer is
//!
//! ```text
//! {"index": u32, "kind": "MAX"|"MIN"|"PROVE"|"CONST"|"CANFAIL",
//!  "term": [node id, ...], "epoch": u32,
//!  "condition": bool term id,
//!  "layer": "syntactic"|"interval"|"z3", "verdict": str,
//!  "value": decimal string | null, "tight": bool,
//!  "interval_bound": decimal string | null,   // MAX/MIN: the bound the ranges alone give
//!  "evidence": {...},
//!  "millis": u64, "cone": {"constraints": u32, "variables": u32}}
//! ```

use serde_json::{json, Value};

use crate::phase1::cost::Primitive;
use crate::phase1::graph::{Form, Graph};
use crate::phase1::interface::{
    Batch, Coordinate, Interface, LayoutKind, LineageStep, Live, RangeAxis,
};
use crate::phase1::oracle::answers::{Answer, Answers, Evidence};
use crate::phase1::oracle::stats::{Stats, Tally};
use crate::phase1::Phase1Result;

/// The schema tag; bump it when the layout above changes.
pub const SCHEMA: &str = "zinnia.phase1/5";

pub fn to_json(result: &Phase1Result) -> Value {
    json!({
        "schema": SCHEMA,
        "requirements": requirements(&result.graph),
        "implementations": implementations(&result.graph),
        "roots": result.graph.roots.iter().map(|at| at.0).collect::<Vec<_>>(),
        "answers": answers(&result.answers),
        "terms": terms(result),
        "stats": stats(&result.stats),
        "missing": result.missing.clone(),
        "violations": result
            .violations()
            .iter()
            .map(|violation| violation.to_string())
            .collect::<Vec<_>>(),
    })
}

fn requirements(graph: &Graph) -> Vec<Value> {
    graph
        .requirements
        .iter()
        .enumerate()
        .map(|(at, node)| {
            let (form, target, interface, family) = match &node.id.form {
                Form::Represent { target, interface } => (
                    "represent",
                    target.render(),
                    Some(interface.render()),
                    Some(interface.family().as_str()),
                ),
                Form::Fetch { value, batch } => (
                    "fetch",
                    value.encode(),
                    Some(format!("Values[Q = {}]", batch.len())),
                    Some("Values"),
                ),
                Form::Enforce { target } => (
                    "enforce",
                    format!("constraint {}", target.constraint()),
                    None,
                    None,
                ),
            };
            let layout = match &node.id.form {
                Form::Represent { interface, .. } => interface_layout(interface),
                _ => Value::Null,
            };
            let batch = match &node.id.form {
                Form::Fetch { batch, .. } => batch_json(batch),
                _ => Value::Null,
            };
            json!({
                "index": at,
                "form": form,
                "target": target,
                "interface": interface,
                "family": family,
                "condition": node.id.condition.index(),
                "epoch": node.id.epoch,
                "sharing": node.id.sharing.0,
                "filter": {"epoch_below": node.id.filter.epoch_below},
                "provisional": node.id.provisional,
                "providers": node.providers.iter().map(|at| at.0).collect::<Vec<_>>(),
                "layout": layout,
                "batch": batch,
            })
        })
        .collect()
}

/// λ = (layout, root, lineage) of a Store-family interface
/// (`rules/README.md` §4.1), machine-readable so that a harness can tell a
/// packed bundle from a masked one without parsing [`Interface::render`].
/// `null` for a family that carries no λ.
fn interface_layout(interface: &Interface) -> Value {
    let Some(lambda) = interface.lambda() else {
        return Value::Null;
    };
    let (grid, ranks) = match &lambda.layout {
        LayoutKind::Box(grid) => (
            Value::Array(grid.iter().map(|n| Value::String(n.to_string())).collect()),
            Value::Null,
        ),
        LayoutKind::Ranked { ranks, .. } => (
            Value::Null,
            ranks
                .as_ref()
                .map(|key| Value::String(key.encode()))
                .unwrap_or(Value::Null),
        ),
        _ => (Value::Null, Value::Null),
    };
    json!({
        "kind": lambda.layout.tag(),
        "slots": lambda.layout.slots().to_string(),
        "grid": grid,
        "ranks": ranks,
        "dense": lambda.layout.is_dense(),
        "ordered": lambda.layout.is_ordered(),
        "root": lambda.root.encode(),
        "lineage": lambda.lineage.iter().map(lineage_step).collect::<Vec<_>>(),
    })
}

/// One position-changing step of λ.lineage (`rules/README.md` §4.1).
fn lineage_step(step: &LineageStep) -> Value {
    match step {
        LineageStep::Filter { mask } => json!({"kind": "filter", "mask": mask.encode()}),
        LineageStep::Slice { lo, hi, step, axis } => json!({
            "kind": "slice",
            "lo": lo.index(),
            "hi": hi.index(),
            "by": step.to_string(),
            "axis": axis,
        }),
        LineageStep::Take { index, axis } => {
            json!({"kind": "take", "index": index.encode(), "axis": axis})
        }
        LineageStep::Transpose { permutation } => {
            json!({"kind": "transpose", "permutation": permutation})
        }
        LineageStep::Reshape { shape } => json!({
            "kind": "reshape",
            "shape": shape.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
        }),
    }
}

/// The batch of a `Fetch` (`rules/README.md` §3.3): the per-axis template of
/// a range batch, or the entry count alone for a general one.
fn batch_json(batch: &Batch) -> Value {
    let axes = match batch {
        Batch::Range { axes, .. } => {
            Value::Array(axes.iter().map(range_axis).collect::<Vec<_>>())
        }
        _ => Value::Null,
    };
    json!({
        "kind": match batch {
            Batch::Scalar => "scalar",
            Batch::General(_) => "general",
            Batch::Range { .. } => "range",
        },
        "entries": batch.len().to_string(),
        "rank": batch.rank(),
        "live": live(batch.live()),
        "axes": axes,
    })
}

fn range_axis(axis: &RangeAxis) -> Value {
    json!({
        "base": coordinate(axis.base),
        "stride": axis.stride.to_string(),
        "width": axis.width.to_string(),
        "broadcast": axis.broadcast.map(|b| b.index()),
    })
}

fn coordinate(coordinate: Coordinate) -> Value {
    match coordinate {
        Coordinate::Literal(value) => json!({"kind": "literal", "value": value.to_string()}),
        Coordinate::Term(term) => json!({"kind": "term", "term": term.index()}),
    }
}

/// What decides whether an entry counts in a run: the default template, or
/// the Boolean term the consumer supplied.
fn live(live: Live) -> Value {
    match live {
        Live::InRange => Value::String("in_range".to_string()),
        Live::Term(term) => json!(term.index()),
    }
}

fn implementations(graph: &Graph) -> Vec<Value> {
    graph
        .implementations
        .iter()
        .enumerate()
        .map(|(at, node)| {
            let parameters: serde_json::Map<String, Value> = node
                .parameters
                .iter()
                .map(|(name, value)| (name.clone(), parameter(*value)))
                .collect();
            json!({
                "index": at,
                "rule": node.rule.as_str(),
                "variant": node.variant,
                "matched": node.matched.iter().map(|key| key.encode()).collect::<Vec<_>>(),
                "inputs": node.inputs.iter().map(|at| at.0).collect::<Vec<_>>(),
                "outputs": node.outputs.iter().map(|at| at.0).collect::<Vec<_>>(),
                "parameters": parameters,
                "epoch": node.epoch,
                "cost": node.cost.to_string(),
                "primitives": node
                    .primitives
                    .iter()
                    .map(|&(primitive, times)| primitive_json(primitive, times))
                    .collect::<Vec<_>>(),
                "fops": node
                    .fops()
                    .iter()
                    .map(|&(name, times)| json!({"op": name, "times": times.to_string()}))
                    .collect::<Vec<_>>(),
                "dropped": node.dropped,
                "readers": node
                    .readers
                    .iter()
                    .map(|&(at, queries)| json!({"requirement": at.0,
                                                 "queries": queries.to_string()}))
                    .collect::<Vec<_>>(),
                "answers": node.answers.iter().map(|at| at.0).collect::<Vec<_>>(),
            })
        })
        .collect()
}

/// One row of an emitter's primitive count (`rules/README.md` §7).
///
/// `name` is the row of §7's table and `times` how often the emitter counts
/// it; the remaining fields are the parameters that row carries, so that a
/// reader can recompute the unit cost of §7 from the JSON without the Rust
/// [`crate::phase1::cost::Cost`] implementation.
fn primitive_json(primitive: Primitive, times: i128) -> Value {
    let times = times.to_string();
    match primitive {
        Primitive::Lin => json!({"name": "lin", "times": times}),
        Primitive::Mul => json!({"name": "mul", "times": times}),
        Primitive::Bool => json!({"name": "bool", "times": times}),
        Primitive::And => json!({"name": "and", "times": times}),
        Primitive::Or => json!({"name": "or", "times": times}),
        Primitive::Xor => json!({"name": "xor", "times": times}),
        Primitive::Not => json!({"name": "not", "times": times}),
        Primitive::Select => json!({"name": "select", "times": times}),
        Primitive::Eq => json!({"name": "eq", "times": times}),
        Primitive::Cmp { width } => json!({"name": "cmp", "times": times, "width": width}),
        Primitive::Range { width } => json!({"name": "range", "times": times, "width": width}),
        Primitive::DivModConst { quotient_width, divisor_width, power_of_two } => json!({
            "name": "divmod_k",
            "times": times,
            "quotient_width": quotient_width,
            "divisor_width": divisor_width,
            "power_of_two": power_of_two,
        }),
        Primitive::DivModWire { width } => {
            json!({"name": "divmod_d", "times": times, "width": width})
        }
        Primitive::Wrap { width, carry_width } => json!({
            "name": "wrap",
            "times": times,
            "width": width,
            "carry_width": carry_width,
        }),
        Primitive::Therm { capacity } => json!({
            "name": "therm", "times": times, "capacity": capacity.to_string(),
        }),
        Primitive::OneHot { capacity } => json!({
            "name": "onehot", "times": times, "capacity": capacity.to_string(),
        }),
        Primitive::Perm { capacity, arity } => json!({
            "name": "perm", "times": times, "capacity": capacity.to_string(), "arity": arity,
        }),
        Primitive::Lookup => json!({"name": "lookup", "times": times}),
        Primitive::Float(op) => json!({"name": "fop", "times": times, "op": op.as_str()}),
        Primitive::Hash { cells } => {
            json!({"name": "hash", "times": times, "cells": cells.to_string()})
        }
        Primitive::Cell => json!({"name": "cell", "times": times}),
        Primitive::Witness => json!({"name": "witness", "times": times}),
    }
}

/// One parameter: an integer as a decimal string, or the range phase 2
/// chooses within (`graph::Parameter`).
fn parameter(value: crate::phase1::graph::Parameter) -> Value {
    match value {
        crate::phase1::graph::Parameter::Fixed(fixed) => Value::String(fixed.to_string()),
        crate::phase1::graph::Parameter::Ranged { lo, hi } => {
            json!({"lo": lo.to_string(), "hi": hi.to_string()})
        }
    }
}

fn answers(answers: &Answers) -> Vec<Value> {
    answers
        .entries()
        .iter()
        .enumerate()
        .map(|(at, answer)| {
            json!({
                "index": at,
                "kind": answer.kind().as_str(),
                "term": answer.key.question.nodes(),
                "epoch": answer.key.epoch,
                "condition": condition(answer.key.condition),
                "layer": answer.layer.as_str(),
                "verdict": answer.verdict.as_str(),
                "value": answer.verdict.value().map(|value| value.to_string()),
                "tight": answer.tight,
                "interval_bound": answer.interval_bound.map(|value| value.to_string()),
                "evidence": evidence(answer),
                "millis": answer.millis as u64,
                "cone": {
                    "constraints": answer.cone_constraints,
                    "variables": answer.cone_variables,
                },
            })
        })
        .collect()
}

/// One oracle condition: the id of the Boolean term it is
/// (`oracle::questions::Condition`).
fn condition(condition: crate::phase1::oracle::questions::Condition) -> Value {
    json!(condition.index())
}

/// The nodes phase 1 built, from phase 0's `node_count()` upward, in phase
/// 0's node format, so that every term id in A and in a condition resolves
/// (`../README.md` §1.1).
fn terms(result: &Phase1Result) -> Value {
    let nodes: Vec<Value> = (result.base_nodes..result.terms.node_count())
        .map(|at| result.terms.node_json(at))
        .collect();
    json!({"base": result.base_nodes, "nodes": nodes})
}

fn evidence(answer: &Answer) -> Value {
    match &answer.evidence {
        Evidence::Syntactic(why) => json!({"kind": "syntactic", "why": why}),
        Evidence::Interval { lo, hi } => json!({
            "kind": "interval",
            "lo": lo.map(|value| value.to_string()),
            "hi": hi.map(|value| value.to_string()),
        }),
        Evidence::Unsat => json!({"kind": "unsat"}),
        Evidence::Model(model) => json!({"kind": "model", "model": model}),
        Evidence::Unknown(why) => json!({"kind": "unknown", "why": why}),
    }
}

fn tally(tally: &Tally) -> Value {
    json!({
        "count": tally.count,
        "total": tally.total.to_string(),
        "max": tally.max.to_string(),
        "mean": tally.mean(),
    })
}

fn stats(stats: &Stats) -> Value {
    let kinds: serde_json::Map<String, Value> = stats
        .kinds
        .iter()
        .map(|(kind, counts)| {
            (
                (*kind).to_string(),
                json!({
                    "asked": counts.asked,
                    "memo_hits": counts.memo_hits,
                    "syntactic": counts.syntactic,
                    "interval": counts.interval,
                    "z3": counts.solver,
                }),
            )
        })
        .collect();
    json!({
        "kinds": kinds,
        "questions": stats.questions(),
        "without_solver": stats.without_solver(),
        "requirements": stats.requirements,
        "implementations": stats.implementations,
        "pruned": stats.pruned,
        "roots": stats.roots,
        "rows": stats.rows,
        "terms_built": stats.terms_built,
        "batch_coordinates": stats.batch_coordinates,
        "constant_coordinates": stats.constant_coordinates,
        "constant_share": stats.constant_share(),
        "solver": {
            "queries": stats.solver.queries,
            "sat": stats.solver.sat,
            "unsat": stats.solver.unsat,
            "unknown": stats.solver.unknown,
            "millis": stats.solver.millis.to_string(),
            "solvers_built": stats.solver.solvers_built,
            "solver_reuse": stats.solver.solver_reuse,
            "sweep_queries": stats.solver.sweep_queries,
            "sweep_millis": stats.solver.sweep_millis.to_string(),
            "sweep_asserted": stats.solver.sweep_asserted,
            "phi_constraints": stats.solver.phi_constraints,
            "phi_variables": stats.solver.phi_variables,
            "cone_constraints": tally(&stats.solver.cone_constraints),
            "cone_variables": tally(&stats.solver.cone_variables),
            "query_millis": tally(&stats.solver.query_millis),
        },
    })
}
