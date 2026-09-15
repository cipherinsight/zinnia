//! The JSON form of a [`Phase0Result`], for the Python harnesses of
//! `acceptance.md` §2.3 and §3.
//!
//! Every value the walk keys has an entry, with the producer that made it;
//! every check and every obligation names the value it belongs to. The
//! invariants those two carry are `implementation.md` §5a's, and
//! `testing/phase0/record.py` checks them on the JSON.
//!
//! # Schema
//!
//! The top-level object is
//!
//! ```text
//! {
//!   "schema":      "zinnia.phase0/2",
//!   "config":      {"loop_limit": u32, "recursion_limit": u32,
//!                   "work_budget": u64},   // the parameters of S the pass read
//!   "terms":       {"vars": [...], "nodes": [...]},   // the term DAG
//!   "variables":   [...],                             // V of Φ
//!   "constraints": [...],                             // C of Φ
//!   "products":    [...],                             // σ = ∏ dims facts
//!   "descriptors": [...],                             // Δ, in traversal order
//!   "delivery":    [key, ...],                        // R
//!   "obligations": [...],                             // CONST / PROVE / CANFAIL
//!   "smtlib":      "..."                              // Φ as SMT-LIB 2 text
//! }
//! ```
//!
//! `terms.nodes[i].id == i`, every argument of a node is a smaller id, and
//! every integer constant is a decimal *string* so that values outside the
//! JSON-safe range survive exactly. `terms.vars[i].id == i` is the `VarId`
//! that `variables[i].id`, `origin`, `var` and `node.var` all refer to.
//!
//! An element of `variables`:
//!
//! ```text
//! {"id": u32, "name": str, "sort": "Idx"|"I64"|"Bool",
//!  "lo": term id | null, "lo_pretty": str | null,
//!  "hi": term id | null, "hi_pretty": str | null,
//!  "origin": {"kind": "input_length"|"count"|"data_scalar"|"data_bool"
//!                     |"shape_symbol"|"unique_count"|"value_sized"|"size_product",
//!             ...}}
//! ```
//!
//! An element of `constraints` is `{"formula": bool term id, "pretty": str,
//! "kind": "var_range"|"check"|"precondition"|"declaration",
//! "provenance": str}` plus
//! `"var"` for a range and `"path"` / `"defcond"` / `"subject"` for a check.
//!
//! A value is named in two ways. Where the record *defines* one — an entry of
//! `descriptors`, a member of `delivery` — the full key object
//! [`ValueKey::to_json`] produces is written, whose `"key"` field is the
//! canonical string `acceptance.md` §2.2 fixes. Where the record *refers* to
//! one — a producer's `alias` and `operands`, a check's or an obligation's
//! `subject` — only that canonical string is written: the components are
//! already in the entry it resolves to, and an unrolled circuit has several
//! references per entry, so the objects would be most of the JSON.
//!
//! An element of `descriptors` is `{"key": key, "name": str | null, "desc":
//! desc, "producer": producer}`, where a *producer* is
//!
//! ```text
//! {"kind": "input"|"alias"|"opaque"|"applied",
//!  "alias": key string | null, "rules": [str] | null,
//!  "operands": [key string] | null, "condition": bool term id | null}
//! ```
//!
//! and a *desc* is one of
//!
//! ```text
//! {"kind": "array",     "dtype": str, "rank": usize,
//!  "dims": [term id],  "dims_pretty": [str], "defcond": id, "psi": id}
//! {"kind": "scalar",    "dtype": str, "origin": "shape"|"data",
//!  "term": {"sort": "int"|"bool", "id": id} | null, "term_pretty": str | null,
//!  "defcond": id, "psi": id}
//! {"kind": "composite", "composite": "list"|"tuple", "elems": [desc]}
//! {"kind": "class",     "type": str}
//! {"kind": "none"}
//! ```
//!
//! An element of `obligations` is `{"kind": "CONST"|"PROVE"|"CANFAIL",
//! "term": {"sort": "int"|"bool", "id": id}, "pretty": str, "path": bool
//! term id, "path_pretty": str, "subject": key string, "provenance": str}`.
//! MAX and MIN are asked by phase 1 and are not recorded (`definition.md`
//! §1).
//!
//! # Pretty strings
//!
//! Every `*_pretty` field is a convenience: the term DAG carries the same
//! information and the SMT-LIB text names every node once. They are written
//! when [`ExportOptions::pretty`] is set, which is the default, and left out
//! when it is not — the pretty form of a term is a tree, so a record whose
//! terms share heavily is far larger with them than without.
//!
//! `products` carries the facts of §3.4 that the term language cannot
//! express, `{"var": u32, "factors": [term id], "pretty": str,
//! "provenance": str}`. They are deliberately absent from `smtlib`, which
//! stays in linear integer arithmetic; the SMT text names them in a trailing
//! comment.

use crate::phase0::desc::{Desc, ScalarTerm};
use crate::phase0::phi::{Constraint, ConstraintKind, Obligation, ObligationTerm};
use crate::phase0::term::{BoolId, TermArena, TermId};
use crate::phase0::Phase0Result;

/// The schema tag; bump it when the layout above changes.
pub const SCHEMA: &str = "zinnia.phase0/2";

/// What to put in the JSON besides the record itself.
#[derive(Debug, Clone, Copy)]
pub struct ExportOptions {
    /// Write the `*_pretty` fields of terms, constraints, obligations and
    /// descriptors.
    pub pretty: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions { pretty: true }
    }
}

/// The pretty printer of one export, which is a no-op when the caller asked
/// for no pretty strings.
struct Pretty<'a> {
    terms: &'a TermArena,
    enabled: bool,
}

impl Pretty<'_> {
    fn int(&self, t: TermId) -> serde_json::Value {
        match self.enabled {
            true => self.terms.pretty(t).into(),
            false => serde_json::Value::Null,
        }
    }

    fn bool(&self, b: BoolId) -> serde_json::Value {
        match self.enabled {
            true => self.terms.pretty_bool(b).into(),
            false => serde_json::Value::Null,
        }
    }

    fn text(&self, make: impl FnOnce() -> String) -> serde_json::Value {
        match self.enabled {
            true => make().into(),
            false => serde_json::Value::Null,
        }
    }
}

/// Render a [`Phase0Result`] as the JSON documented above.
pub fn to_json(result: &Phase0Result, options: &ExportOptions) -> serde_json::Value {
    let terms = &result.terms;
    let pretty = Pretty {
        terms,
        enabled: options.pretty,
    };
    serde_json::json!({
        "schema": SCHEMA,
        // Every compilation parameter the pass read, so that a record says
        // which budget produced it: `Reject::BudgetExceeded` and
        // `E-ValueSized`'s maximum both depend on `work_budget`.
        "config": {
            "loop_limit": result.config.loop_limit,
            "recursion_limit": result.config.recursion_limit,
            "work_budget": result.config.work_budget,
        },
        "terms": terms.to_json(),
        "variables": variables_json(result, &pretty),
        "constraints": result
            .phi
            .constraints()
            .iter()
            .map(|c| constraint_json(&pretty, c))
            .collect::<Vec<_>>(),
        "products": products_json(result, &pretty),
        "descriptors": result
            .descriptors
            .iter()
            .map(|entry| serde_json::json!({
                "key": entry.key.to_json(),
                "name": entry.name,
                "desc": desc_json(&pretty, &entry.desc),
                "producer": entry.producer.to_json(),
            }))
            .collect::<Vec<_>>(),
        "delivery": result.delivery.iter().map(|k| k.to_json()).collect::<Vec<_>>(),
        "obligations": result
            .obligations()
            .iter()
            .map(|o| obligation_json(&pretty, o))
            .collect::<Vec<_>>(),
        "smtlib": smtlib(result),
    })
}

fn variables_json(result: &Phase0Result, pretty: &Pretty) -> Vec<serde_json::Value> {
    let terms = &result.terms;
    result
        .phi
        .vars()
        .iter()
        .enumerate()
        .map(|(i, info)| {
            let decl = terms.var_decl(crate::phase0::term::VarId(i as u32));
            serde_json::json!({
                "id": i,
                "name": decl.name,
                "sort": decl.sort.as_str(),
                "lo": info.lo.map(|t| t.index()),
                "lo_pretty": info.lo.map(|t| pretty.int(t)),
                "hi": info.hi.map(|t| t.index()),
                "hi_pretty": info.hi.map(|t| pretty.int(t)),
                "origin": info.origin.to_json(),
            })
        })
        .collect()
}

fn products_json(result: &Phase0Result, pretty: &Pretty) -> Vec<serde_json::Value> {
    let terms = &result.terms;
    result
        .phi
        .products()
        .iter()
        .map(|p| {
            serde_json::json!({
                "var": p.var.0,
                "factors": p.factors.iter().map(|t| t.index()).collect::<Vec<_>>(),
                "pretty": pretty.text(|| {
                    let factors: Vec<String> =
                        p.factors.iter().map(|&t| terms.pretty(t)).collect();
                    format!("{} = {}", terms.var_name(p.var), factors.join(" * "))
                }),
                "provenance": p.provenance,
            })
        })
        .collect()
}

fn constraint_json(pretty: &Pretty, c: &Constraint) -> serde_json::Value {
    let mut value = serde_json::json!({
        "formula": c.formula.index(),
        "pretty": pretty.bool(c.formula),
        "provenance": c.provenance,
    });
    let object = value.as_object_mut().expect("json! built an object");
    match &c.kind {
        ConstraintKind::VarRange(v) => {
            object.insert("kind".into(), "var_range".into());
            object.insert("var".into(), v.0.into());
        }
        ConstraintKind::Check {
            path,
            defcond,
            subject,
        } => {
            object.insert("kind".into(), "check".into());
            object.insert("path".into(), path.index().into());
            object.insert("defcond".into(), defcond.index().into());
            object.insert("subject".into(), subject.encode().into());
        }
        ConstraintKind::Precondition => {
            object.insert("kind".into(), "precondition".into());
        }
        ConstraintKind::Declaration => {
            object.insert("kind".into(), "declaration".into());
        }
    }
    value
}

fn obligation_json(pretty: &Pretty, o: &Obligation) -> serde_json::Value {
    let (term, text) = match o.term {
        ObligationTerm::Int(t) => (
            serde_json::json!({"sort": "int", "id": t.index()}),
            pretty.int(t),
        ),
        ObligationTerm::Bool(b) => (
            serde_json::json!({"sort": "bool", "id": b.index()}),
            pretty.bool(b),
        ),
    };
    serde_json::json!({
        "kind": o.kind.as_str(),
        "term": term,
        "pretty": text,
        "path": o.path.index(),
        "path_pretty": pretty.bool(o.path),
        "subject": o.subject.encode(),
        "provenance": o.provenance,
    })
}

fn desc_json(pretty: &Pretty, desc: &Desc) -> serde_json::Value {
    match desc {
        Desc::Array(a) => serde_json::json!({
            "kind": "array",
            "dtype": a.dtype.as_str(),
            "rank": a.rank(),
            "dims": a.dims.iter().map(|t| t.index()).collect::<Vec<_>>(),
            "dims_pretty": a.dims.iter().map(|&t| pretty.int(t)).collect::<Vec<_>>(),
            "defcond": a.defcond.index(),
            "psi": a.psi.index(),
        }),
        Desc::Scalar(s) => {
            let (term, text) = match s.term {
                ScalarTerm::Int(t) => (
                    Some(serde_json::json!({"sort": "int", "id": t.index()})),
                    pretty.int(t),
                ),
                ScalarTerm::Bool(b) => (
                    Some(serde_json::json!({"sort": "bool", "id": b.index()})),
                    pretty.bool(b),
                ),
                ScalarTerm::Opaque => (None, serde_json::Value::Null),
            };
            serde_json::json!({
                "kind": "scalar",
                "dtype": s.dtype.as_str(),
                "origin": s.origin.as_str(),
                "term": term,
                "term_pretty": text,
                "defcond": s.defcond.index(),
                "psi": s.psi.index(),
            })
        }
        Desc::Composite(c) => serde_json::json!({
            "kind": "composite",
            "composite": c.kind.as_str(),
            "elems": c.elems.iter().map(|e| desc_json(pretty, e)).collect::<Vec<_>>(),
        }),
        Desc::Class(t) => serde_json::json!({"kind": "class", "type": t.to_string()}),
        Desc::None => serde_json::json!({"kind": "none"}),
    }
}

/// $\Phi$ as SMT-LIB 2 text: the variables, every term of the arena as a
/// named definition, and one assert per constraint. The product facts of
/// §3.4 are nonlinear, so they appear only as a trailing comment; phase 1
/// bounds each factor separately and multiplies.
fn smtlib(result: &Phase0Result) -> String {
    let asserts: Vec<_> = result
        .phi
        .constraints()
        .iter()
        .map(|c| (c.formula, c.provenance.clone()))
        .collect();
    let mut text = result.terms.to_smtlib(&asserts);
    for p in result.phi.products() {
        let factors: Vec<String> = p.factors.iter().map(|&t| result.terms.pretty(t)).collect();
        text.push_str(&format!(
            "; product fact (definition.md 3.4): {} = {}\n",
            result.terms.var_name(p.var),
            factors.join(" * ")
        ));
    }
    text
}
