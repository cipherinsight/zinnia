//! Fixtures for the phase 1 tests: the AST JSON the front end would export,
//! built node by node, and the helpers that run phase 0 and phase 1 over it.
//!
//! The builders mirror `src/phase0/tests/fixtures.rs`, so a program is
//! written here the way it is written there and the two suites read alike.
//! A phase 1 test needs the `ASTCircuit` after the run — the AST index reads
//! it — so [`Program`] owns it.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::phase0::{analyze, Phase0Config, Phase0Result};
use crate::phase1::cost::{Cost, UnitCost};
use crate::phase1::omega::Omega;
use crate::phase1::oracle::Limits;
use crate::phase1::{build_with, Phase1Result, Reject};

// ── AST builders ────────────────────────────────────────────────────

pub(super) fn load(name: &str) -> Value {
    json!({"__class__": "ASTLoad", "name": name})
}

pub(super) fn int(value: i64) -> Value {
    json!({"__class__": "ASTConstantInteger", "value": value})
}

pub(super) fn binop(operator: &str, lhs: Value, rhs: Value) -> Value {
    json!({"__class__": "ASTBinaryOperator", "operator": operator, "lhs": lhs, "rhs": rhs})
}

pub(super) fn call(target: Option<&str>, member: &str, args: Vec<Value>) -> Value {
    json!({"__class__": "ASTNamedAttribute", "target": target, "member": member,
           "args": args, "kwargs": {}})
}

pub(super) fn name_target(name: &str) -> Value {
    json!({"__class__": "ASTNameAssignTarget", "name": name, "star": false})
}

pub(super) fn assign(name: &str, value: Value) -> Value {
    json!({"__class__": "ASTAssignStatement", "targets": [name_target(name)], "value": value})
}

pub(super) fn tuple(values: Vec<Value>) -> Value {
    json!({"__class__": "ASTParenthesis", "values": values})
}

/// `a, b = e`, an unpacking assignment.
pub(super) fn assign_tuple(names: &[&str], value: Value) -> Value {
    let targets: Vec<Value> = names.iter().map(|name| name_target(name)).collect();
    json!({"__class__": "ASTAssignStatement",
           "targets": [{"__class__": "ASTTupleAssignTarget", "targets": targets, "star": false}],
           "value": value})
}

/// `for _ in range(n): body`, a loop whose trip count is a bounded scalar.
pub(super) fn for_range(bound: &str, body: Vec<Value>) -> Value {
    json!({"__class__": "ASTForInStatement",
           "target": name_target("_"),
           "iter_expr": call(None, "range", vec![load(bound)]),
           "block": body, "orelse": []})
}

/// `v[entries]`: a subscript whose entries are index nodes, `lo:hi:step`
/// triples and `np.newaxis` markers, in source order.
pub(super) fn subscript(value: Value, entries: Vec<Value>) -> Value {
    json!({"__class__": "ASTSubscriptExp", "val": value,
           "slicing": {"__class__": "ASTSlice", "data": entries}})
}

/// `lo:hi:step` as a subscript entry; `None` is an omitted endpoint, which
/// the front end exports as JSON `null`.
pub(super) fn slice_range(lo: Option<Value>, hi: Option<Value>, step: Option<Value>) -> Value {
    json!([lo, hi, step])
}

/// `np.newaxis` as a subscript entry.
pub(super) fn new_axis() -> Value {
    json!({"__class__": "ASTSliceNewAxis"})
}

/// A call with keyword arguments, such as `np.sum(a, axis=1)`.
pub(super) fn call_kw(
    target: Option<&str>,
    member: &str,
    args: Vec<Value>,
    kwargs: Vec<(&str, Value)>,
) -> Value {
    let kw: serde_json::Map<String, Value> = kwargs
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect();
    json!({"__class__": "ASTNamedAttribute", "target": target, "member": member,
           "args": args, "kwargs": kw})
}

/// `x.m(args)`, a method call on an expression.
pub(super) fn method(target: Value, member: &str, args: Vec<Value>) -> Value {
    json!({"__class__": "ASTExprAttribute", "target": target, "member": member,
           "args": args, "kwargs": {}})
}

/// A tuple display of integer literals, for an `axes` or `shape` argument.
pub(super) fn ints(values: &[i64]) -> Value {
    tuple(values.iter().map(|&value| int(value)).collect())
}

/// `v[entries] = value`, a subscript assignment.
pub(super) fn assign_subscript(name: &str, entries: Vec<Value>, value: Value) -> Value {
    json!({"__class__": "ASTAssignStatement",
           "targets": [{"__class__": "ASTSubscriptAssignTarget",
                        "target": load(name),
                        "slicing": {"__class__": "ASTSlice", "data": entries},
                        "star": false}],
           "value": value})
}

/// The Boolean literal `True`, for a keyword flag such as `return_inverse`.
pub(super) fn json_true() -> Value {
    json!({"__class__": "ASTConstantBoolean", "value": true})
}

pub(super) fn ret(expr: Value) -> Value {
    json!({"__class__": "ASTReturnStatement", "expr": expr})
}

pub(super) fn assert_stmt(expr: Value) -> Value {
    json!({"__class__": "ASTAssertStatement", "expr": expr})
}

pub(super) fn cond(test: Value, then: Vec<Value>, otherwise: Vec<Value>) -> Value {
    json!({"__class__": "ASTCondStatement", "cond": test,
           "t_block": then, "f_block": otherwise})
}

pub(super) fn scalar_input(name: &str) -> Value {
    json!({"name": name, "annotation": {"dt": "Integer", "kind": "Private"}})
}


/// A tuple-typed parameter, `Tuple[…]`, whose elements Δ keys one by one
/// (`../../phase0/definition.md` §3.3).
pub(super) fn tuple_input(name: &str, elements: &[&str]) -> Value {
    let elems: Vec<Value> = elements.iter().map(|dt| json!(dt)).collect();
    json!({"name": name,
           "annotation": {"dt": {"Tuple": {"elements": elems}}, "kind": "Private"}})
}

/// A fixed-shape `Float` array parameter, `Float[Array, "…"]`.
pub(super) fn float_array_input(name: &str, shape: &[usize]) -> Value {
    json!({"name": name,
           "annotation": {"dt": {"NDArray": {"shape": shape, "dtype": "Float"}},
                          "kind": "Private"}})
}


pub(super) fn array_input(name: &str, shape: &[usize]) -> Value {
    json!({"name": name,
           "annotation": {"dt": {"NDArray": {"shape": shape, "dtype": "Integer"}},
                          "kind": "Private"}})
}

/// A float literal.
pub(super) fn float(value: f64) -> Value {
    json!({"__class__": "ASTConstantFloat", "value": value})
}

/// A variable-length `Float` array, `Float[Array, "N<=bound"]`.
pub(super) fn float_varlen_input(name: &str, symbol: &str, bound: usize) -> Value {
    json!({"name": name,
           "annotation": {"dt": {"SymbolicNDArray": {
               "shape": [{"symbol": symbol, "upper": bound}],
               "dtype": "Float", "bindings": {}}},
               "kind": "Private"}})
}

/// A variable-length integer array, `Int[Array, "N<=bound"]`.
pub(super) fn varlen_input(name: &str, symbol: &str, bound: usize) -> Value {
    json!({"name": name,
           "annotation": {"dt": {"SymbolicNDArray": {
               "shape": [{"symbol": symbol, "upper": bound}],
               "dtype": "Integer", "bindings": {}}},
               "kind": "Private"}})
}

/// A rank-2 integer array whose two axes are both variable-length,
/// `Int[Array, "N<=n M<=m"]`, which is what gives a size a product fact.
pub(super) fn varlen_input_2d(name: &str, axes: &[(&str, usize)]) -> Value {
    let shape: Vec<Value> = axes
        .iter()
        .map(|(symbol, upper)| json!({"symbol": symbol, "upper": upper}))
        .collect();
    json!({"name": name,
           "annotation": {"dt": {"SymbolicNDArray": {
               "shape": shape, "dtype": "Integer", "bindings": {}}},
               "kind": "Private"}})
}

/// `@requires(lambda x: lo <= x <= hi)` as the front end exports it.
pub(super) fn requires_between(name: &str, lo: i64, hi: i64) -> Value {
    let var = json!({"Var": {"Input": name}});
    json!({"term": {"BoolComb": {"op": "And", "operands": [
        {"Cmp": {"op": "Le", "lhs": {"LitInt": lo}, "rhs": var}},
        {"Cmp": {"op": "Le", "lhs": var, "rhs": {"LitInt": hi}}}
    ]}}})
}

/// `@requires(lambda a, b: a <= b)`, a fact over two parameters.
pub(super) fn requires_le(left: &str, right: &str) -> Value {
    json!({"term": {"Cmp": {"op": "Le",
                            "lhs": {"Var": {"Input": left}},
                            "rhs": {"Var": {"Input": right}}}}})
}

// ── Running the two phases ──────────────────────────────────────────

/// One program with both phases run over it.
pub(super) struct Program {
    pub phase0: Phase0Result,
    pub phase1: Phase1Result,
}

impl Program {
    /// Every implementation node of one rule, live in G.
    pub fn nodes_of(&self, rule: crate::phase1::rules::RuleId) -> Vec<&crate::phase1::graph::ImplNode> {
        self.phase1
            .graph
            .live_implementations()
            .into_iter()
            .map(|at| self.phase1.graph.implementation(at))
            .filter(|node| node.rule == rule)
            .collect()
    }

    /// The rules of every live implementation node, sorted, for a shape
    /// assertion that does not depend on the order rows were filled in.
    pub fn rules(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self
            .phase1
            .graph
            .live_implementations()
            .into_iter()
            .map(|at| self.phase1.graph.implementation(at).rule.as_str())
            .collect();
        names.sort();
        names
    }

    /// The graph as lines, for the message of a failing assertion.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        for (at, node) in self.phase1.graph.requirements.iter().enumerate() {
            out.push_str(&format!(
                "R{} {:?} cond {} epoch {:?} excl {:?} providers {:?}\n",
                at,
                node.id.form,
                node.id.condition.index(),
                node.id.epoch,
                node.id.filter,
                node.providers
            ));
        }
        for (at, node) in self.phase1.graph.implementations.iter().enumerate() {
            out.push_str(&format!(
                "I{} {} {:?} epoch {} cost {} in {:?} out {:?}\n",
                at,
                node.rule.as_str(),
                node.variant,
                node.epoch,
                node.cost,
                node.inputs,
                node.outputs
            ));
        }
        out
    }

    /// The answers of one kind in A.
    pub fn answers_of(
        &self,
        kind: crate::phase1::oracle::questions::Kind,
    ) -> Vec<&crate::phase1::oracle::answers::Answer> {
        self.phase1
            .answers
            .entries()
            .iter()
            .filter(|answer| answer.kind() == kind)
            .collect()
    }
}

/// Parse a circuit fixture, run phase 0, then phase 1.
pub(super) fn run_full(
    inputs: Vec<Value>,
    block: Vec<Value>,
    scalar_requires: Vec<Value>,
) -> Result<Program, Reject> {
    run_with(inputs, block, scalar_requires, &Omega::default(), &UnitCost)
}

/// [`run_full`] with a caller-chosen Ω and cost function.
pub(super) fn run_with(
    inputs: Vec<Value>,
    block: Vec<Value>,
    scalar_requires: Vec<Value>,
    omega: &Omega,
    cost: &dyn Cost,
) -> Result<Program, Reject> {
    let ast = json!({"__class__": "ASTCircuit", "block": block, "inputs": inputs,
                     "scalar_requires": scalar_requires});
    let circuit: crate::ast::ASTCircuit =
        serde_json::from_value(ast).expect("the fixture is a valid ASTCircuit");
    let chips: HashMap<String, Value> = HashMap::new();
    let externals: HashMap<String, Value> = HashMap::new();
    let phase0 = analyze(&circuit, &chips, &externals, &Phase0Config::default())
        .expect("phase 0 accepts the fixture");
    let phase1 = build_with(&phase0, &circuit, &chips, omega, cost, Limits::default())?;
    Ok(Program { phase0, phase1 })
}

/// [`run_full`] with no `@requires` facts.
pub(super) fn run(inputs: Vec<Value>, block: Vec<Value>) -> Result<Program, Reject> {
    run_full(inputs, block, Vec::new())
}
