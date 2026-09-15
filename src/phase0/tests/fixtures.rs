//! Fixtures shared by the walker tests: the AST JSON the front end would
//! export, built node by node, and the helpers that run the pass over it and
//! read the result.
//!
//! Every field name is the one `src/ast/mod.rs` declares, so a fixture is the
//! same serde input [`analyze`] sees from `analyze_shapes`.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::phase0::desc::{Desc, ValueKey};
use crate::phase0::{analyze, Phase0Config, Phase0Result, Reject};

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

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

pub(super) fn tuple(values: Vec<Value>) -> Value {
    json!({"__class__": "ASTParenthesis", "values": values})
}

pub(super) fn name_target(name: &str) -> Value {
    json!({"__class__": "ASTNameAssignTarget", "name": name, "star": false})
}

pub(super) fn starred_target(name: &str) -> Value {
    json!({"__class__": "ASTNameAssignTarget", "name": name, "star": true})
}

pub(super) fn assign(target: Value, value: Value) -> Value {
    json!({"__class__": "ASTAssignStatement", "targets": [target], "value": value})
}

pub(super) fn pass_stmt() -> Value {
    json!({"__class__": "ASTPassStatement"})
}

pub(super) fn ret(expr: Value) -> Value {
    json!({"__class__": "ASTReturnStatement", "expr": expr})
}

pub(super) fn cond(test: Value, then: Vec<Value>, otherwise: Vec<Value>) -> Value {
    json!({"__class__": "ASTCondStatement", "cond": test,
           "t_block": then, "f_block": otherwise})
}

pub(super) fn for_in(target: &str, iterable: Value, body: Vec<Value>) -> Value {
    json!({"__class__": "ASTForInStatement", "target": name_target(target),
           "iter_expr": iterable, "block": body, "orelse": []})
}

pub(super) fn while_loop(test: Value, body: Vec<Value>) -> Value {
    json!({"__class__": "ASTWhileStatement", "test_expr": test,
           "block": body, "orelse": []})
}

pub(super) fn subscript(value: Value, index: Value) -> Value {
    json!({"__class__": "ASTSubscriptExp", "val": value,
           "slicing": {"__class__": "ASTSlice", "data": [index]}})
}

/// A subscript with several entries: an index node, a `slice_range`, or a
/// `new_axis` marker, in source order.
pub(super) fn subscript_at(value: Value, entries: Vec<Value>) -> Value {
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

pub(super) fn subscript_target(name: &str, index: Value) -> Value {
    json!({"__class__": "ASTSubscriptAssignTarget", "target": load(name),
           "slicing": {"__class__": "ASTSlice", "data": [index]}, "star": false})
}

/// A fixed-shape integer array input.
pub(super) fn array_input(name: &str, shape: &[usize]) -> Value {
    json!({"name": name,
           "annotation": {"dt": {"NDArray": {"shape": shape, "dtype": "Integer"}}}})
}

/// A scalar integer input, which is a data scalar (§3.2).
pub(super) fn scalar_input(name: &str) -> Value {
    json!({"name": name, "annotation": {"dt": "Integer"}})
}

/// `@requires(lambda x: lo <= x <= hi)` as the front end exports it: one
/// `ContractTerm` of two comparisons (`coverage.md` §4).
pub(super) fn requires_between(name: &str, lo: i64, hi: i64) -> Value {
    let var = json!({"Var": {"Input": name}});
    json!({"term": {"BoolComb": {"op": "And", "operands": [
        {"Cmp": {"op": "Le", "lhs": {"LitInt": lo}, "rhs": var}},
        {"Cmp": {"op": "Le", "lhs": var, "rhs": {"LitInt": hi}}}
    ]}}})
}

/// Run the pass on a circuit with the given inputs, body, chips and config.
pub(super) fn run_with(
    inputs: Vec<Value>,
    block: Vec<Value>,
    chips: HashMap<String, Value>,
    config: Phase0Config,
) -> Result<Phase0Result, Reject> {
    run_full(inputs, block, chips, config, Vec::new())
}

/// [`run_with`] plus the circuit's `scalar_requires` facts.
pub(super) fn run_full(
    inputs: Vec<Value>,
    block: Vec<Value>,
    chips: HashMap<String, Value>,
    config: Phase0Config,
    scalar_requires: Vec<Value>,
) -> Result<Phase0Result, Reject> {
    let ast = json!({"__class__": "ASTCircuit", "block": block, "inputs": inputs,
                     "scalar_requires": scalar_requires});
    let circuit: crate::ast::ASTCircuit =
        serde_json::from_value(ast).expect("the fixture is a valid ASTCircuit");
    let externals = HashMap::new();
    let result = analyze(&circuit, &chips, &externals, &config);
    // Every walk test runs through here, so the record invariants of
    // `implementation.md` §5a are checked on every program the tests walk.
    if let Ok(record) = &result {
        super::invariants::assert_record_invariants(record);
    }
    result
}

pub(super) fn run(inputs: Vec<Value>, block: Vec<Value>) -> Result<Phase0Result, Reject> {
    run_with(inputs, block, HashMap::new(), Phase0Config::default())
}

/// The descriptor of the delivery set (§1's `R`).
pub(super) fn delivered(result: &Phase0Result) -> &Desc {
    let key: &ValueKey = result
        .delivery
        .first()
        .expect("the circuit returns a value");
    result.descriptor(key).expect("R is recorded in Δ")
}

/// The single dimension of a delivered rank-1 array, as a constant.
pub(super) fn delivered_length(result: &Phase0Result) -> Option<i128> {
    let array = delivered(result).as_array().expect("an array was returned");
    result.terms.constant_value(array.dims[0])
}

/// The provenance strings of every check `Φ` carries.
pub(super) fn check_provenances(result: &Phase0Result) -> Vec<String> {
    result
        .phi
        .constraints()
        .iter()
        .filter(|c| matches!(c.kind, crate::phase0::phi::ConstraintKind::Check { .. }))
        .map(|c| c.provenance.clone())
        .collect()
}
