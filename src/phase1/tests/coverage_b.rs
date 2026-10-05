//! The shapes the coverage pass over the reduction, elementwise, scalar,
//! ordering and in-place rules added: one fixture per rule file extended.
//!
//! Each test names the rule file's section it exercises, so that a failure
//! points at the paragraph the node implements.

use serde_json::Value;

use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::rules::RuleId;

use super::fixtures::*;

/// How many times a node emits one primitive.
fn times(node: &crate::phase1::graph::ImplNode, wanted: Primitive) -> i128 {
    node.primitives
        .iter()
        .filter(|&&(primitive, _)| primitive == wanted)
        .map(|&(_, count)| count)
        .sum()
}

/// A Float scalar, read out of a one-element array: the fixtures build a
/// Float value only as an array element.
fn float_scalar(array: &str) -> Value {
    subscript(load(array), vec![int(0)])
}

/// `...` as a subscript entry, as the front end exports it.
fn ellipsis() -> Value {
    serde_json::json!({"__class__": "ASTSliceEllipsis"})
}

// ── elementwise: the rank-0 result ──────────────────────────────────

/// `elementwise_batch.md`, the `scalar` variant: an elementwise operation
/// whose result has rank 0 is a scalar, and its `Wire` row is the one-entry
/// `Values` of `rules/README.md` §3.2.
#[test]
fn a_scalar_elementwise_result_is_one_wire() {
    let program = run(
        vec![float_array_input("x", &[2])],
        vec![
            assign("a", float_scalar("x")),
            ret(call(Some("np"), "sin", vec![load("a")])),
        ],
    )
    .expect("phase 1 accepts an elementwise operation on a scalar");

    let nodes = program.nodes_of(RuleId::ElementwiseBatch);
    let node = nodes
        .iter()
        .find(|node| node.variant == Some("scalar"))
        .unwrap_or_else(|| panic!("the `scalar` variant\n{}", program.dump()));
    assert_eq!(node.fops(), [("sin_f", 1)], "one `sin_f` and nothing else");
    assert_eq!(node.parameter("Q"), Some(1), "the batch is the one entry `[()]`");
}

/// A chain of scalar elementwise operations is one node each, the inner
/// one's `Wire` row the outer one's port — `np.sin(np.exp(a))`, which is the
/// shape `ds1000_case0326` and `pythran_nd_local_maxima::reflect` are.
#[test]
fn a_chain_of_scalar_maps_is_one_node_each() {
    let program = run(
        vec![float_array_input("x", &[2])],
        vec![
            assign("a", float_scalar("x")),
            ret(call(
                Some("np"),
                "sin",
                vec![call(Some("np"), "exp", vec![load("a")])],
            )),
        ],
    )
    .expect("phase 1 accepts a chain of elementwise operations on a scalar");

    let nodes: Vec<_> = program
        .nodes_of(RuleId::ElementwiseBatch)
        .into_iter()
        .filter(|node| node.variant == Some("scalar"))
        .collect();
    assert_eq!(nodes.len(), 2, "one node per operation\n{}", program.dump());
    let mut emitted: Vec<Vec<(&'static str, i128)>> =
        nodes.iter().map(|node| node.fops()).collect();
    emitted.sort();
    assert_eq!(emitted, vec![vec![("exp_f", 1)], vec![("sin_f", 1)]]);
}

/// `map_aligned.md`, the `power` row: `a ** e` for a literal `e ≥ 1` is
/// square-and-multiply, `n_e = ⌊log₂ e⌋ + popcount(e) − 1` products.
#[test]
fn an_integer_power_is_square_and_multiply() {
    let program = run(
        vec![array_input("x", &[4])],
        vec![ret(binop("pow", load("x"), int(3)))],
    )
    .expect("phase 1 accepts an integer power with a literal exponent");

    let nodes = program.nodes_of(RuleId::MapAligned);
    let node = nodes
        .first()
        .unwrap_or_else(|| panic!("MAP_ALIGNED has a `power` row\n{}", program.dump()));
    // e = 3: one squaring and one further product, at four slots.
    assert_eq!(times(node, Primitive::Mul), 2 * 4, "two products per slot");
}

/// A literal exponent of 0 is the constant 1 and emits nothing.
#[test]
fn a_zero_exponent_emits_no_product() {
    let program = run(
        vec![array_input("x", &[4])],
        vec![ret(binop("pow", load("x"), int(0)))],
    )
    .expect("phase 1 accepts `a ** 0`");

    let nodes = program.nodes_of(RuleId::MapAligned);
    let node = nodes
        .first()
        .unwrap_or_else(|| panic!("MAP_ALIGNED has a `power` row\n{}", program.dump()));
    assert_eq!(times(node, Primitive::Mul), 0, "`a ** 0` is the constant 1");
}

// ── reduction: the rank-0 array result ──────────────────────────────

/// `export.md`, `required_input_ports`: a rank-0 array is a scalar, so its
/// `Export` is the one cell of its wire and not a bundle. `np.mean(x,
/// axis=0)` on a rank-1 x delivers one.
#[test]
fn a_rank_zero_array_is_exported_as_one_wire() {
    let program = run(
        vec![float_array_input("x", &[4])],
        vec![ret(call_kw(
            Some("np"),
            "mean",
            vec![load("x")],
            vec![("axis", int(0))],
        ))],
    )
    .expect("phase 1 accepts a rank-0 reduction result");

    let exports = program.nodes_of(RuleId::Export);
    assert!(
        exports.iter().any(|node| node.variant == Some("scalar")),
        "the root is the scalar export\n{}",
        program.dump()
    );
    assert!(
        !program.nodes_of(RuleId::FoldGroups).is_empty(),
        "the fold answers the wire\n{}",
        program.dump()
    );
}

// ── ordering: `ord_τ` for Float ─────────────────────────────────────

/// `ordering/README.md` §2.2, §2.3 and Open 3: a Float operand has no
/// order-preserving map into `[0, 2^β)`, so the order check is the backend's
/// own comparison — `lte_f` for `SORT`.
#[test]
fn a_float_sort_orders_by_comparison() {
    let program = run(
        vec![float_array_input("x", &[4])],
        vec![ret(call(Some("np"), "sort", vec![load("x")]))],
    )
    .expect("phase 1 accepts a Float sort");

    let nodes = program.nodes_of(RuleId::Sort);
    let node = nodes
        .first()
        .unwrap_or_else(|| panic!("SORT matches a Float operand\n{}", program.dump()));
    assert_eq!(
        times(node, Primitive::Float(FloatOp::Lte)),
        3,
        "one comparison per adjacent pair\n{}",
        program.dump()
    );
    assert_eq!(times(node, Primitive::Range { width: 0 }), 0, "no range check");
    assert_eq!(node.parameter("beta"), Some(0), "a Float wire has no width");
}

/// `argsort.md`, emitter step 4: the strict check on the combined key
/// becomes the two-comparison form, with `lt_f` and `eq_f` on the values and
/// a `cmp_b` on the indices.
#[test]
fn a_float_argsort_breaks_ties_on_the_index() {
    let program = run(
        vec![float_array_input("x", &[4])],
        vec![ret(call(Some("np"), "argsort", vec![load("x")]))],
    )
    .expect("phase 1 accepts a Float argsort");

    let nodes = program.nodes_of(RuleId::Argsort);
    let node = nodes
        .first()
        .unwrap_or_else(|| panic!("ARGSORT matches a Float operand\n{}", program.dump()));
    assert_eq!(times(node, Primitive::Float(FloatOp::Lt)), 3, "one `lt_f` per pair");
    assert_eq!(times(node, Primitive::Float(FloatOp::Eq)), 3, "one `eq_f` per pair");
    assert_eq!(node.parameter("b"), Some(3), "b = bits(4), the index width");
}

// ── representation: the entries `SETITEM` reads ─────────────────────

/// `setitem.md`, the I row: an integer entry with a *runtime* index takes
/// its normalised position from `INDEX_NORMALIZE`, whose bundle is attached
/// to the row the consumer asked for.
#[test]
fn a_runtime_row_index_is_normalised() {
    let program = run(
        vec![array_input("a", &[4, 4]), scalar_input("r")],
        vec![
            assign_subscript("a", vec![load("r"), slice_range(None, None, None)], int(0)),
            ret(load("a")),
        ],
    )
    .expect("phase 1 accepts a write at a runtime row index");

    assert!(
        !program.nodes_of(RuleId::IndexNormalize).is_empty(),
        "the `idx_k` port has a provider\n{}",
        program.dump()
    );
    assert!(!program.nodes_of(RuleId::SetItem).is_empty(), "the write is a node");
}

/// `setitem.md`, the F row: an index array beside the full and slice entries
/// of the other axes, on any axis.
#[test]
fn an_index_array_writes_beside_a_slice() {
    let program = run(
        vec![array_input("a", &[4, 4]), array_input("i", &[2])],
        vec![
            assign_subscript("a", vec![slice_range(None, None, None), load("i")], int(0)),
            ret(load("a")),
        ],
    )
    .expect("phase 1 accepts an index array on axis 1 beside a slice");

    assert!(!program.nodes_of(RuleId::SetItem).is_empty(), "the write is a node");
}

/// `setitem.md`, the `full` row: an ellipsis stands for the axes the other
/// entries do not name.
#[test]
fn an_ellipsis_writes_every_axis() {
    let program = run(
        vec![array_input("a", &[4, 4])],
        vec![
            assign_subscript("a", vec![ellipsis()], int(0)),
            ret(load("a")),
        ],
    )
    .expect("phase 1 accepts a write through an ellipsis");

    let nodes = program.nodes_of(RuleId::SetItem);
    let node = nodes
        .first()
        .unwrap_or_else(|| panic!("SETITEM matches an ellipsis\n{}", program.dump()));
    assert_eq!(node.parameter("Q"), Some(16), "every slot of the box is written");
}

// ── reduction: the contractions ─────────────────────────────────────

/// `matmul.md`, `source_pattern`: a left operand of rank ≥ 2 carries its
/// leading axes into the result as free row axes, which is what `a @ b` on a
/// stack of matrices means.
#[test]
fn a_stack_of_matrices_contracts_its_last_axis() {
    let program = run(
        vec![array_input("a", &[2, 3, 4]), array_input("b", &[4, 5])],
        vec![ret(call(Some("np"), "matmul", vec![load("a"), load("b")]))],
    )
    .expect("phase 1 accepts a rank-3 left operand");

    let nodes = program.nodes_of(RuleId::Matmul);
    let node = nodes
        .first()
        .unwrap_or_else(|| panic!("MATMUL matches a rank-3 operand\n{}", program.dump()));
    assert_eq!(node.parameter("K"), Some(4), "the contracted extent");
    assert_eq!(node.parameter("Q"), Some(2 * 3 * 5), "one position per output slot");
}

/// `matmul.md`, the `correlate` form: `c[j] = Σ_{k<W} long[j + k]·short[k]`
/// with `W = min(n, m)` and `P = |n − m| + 1`.
#[test]
fn a_correlation_folds_the_shorter_operand() {
    let program = run(
        vec![array_input("a", &[6]), array_input("b", &[4])],
        vec![ret(call(Some("np"), "correlate", vec![load("a"), load("b")]))],
    )
    .expect("phase 1 accepts a correlation of two constant lengths");

    let nodes = program.nodes_of(RuleId::Matmul);
    let node = nodes
        .iter()
        .find(|node| node.variant == Some("correlate"))
        .unwrap_or_else(|| panic!("the `correlate` variant\n{}", program.dump()));
    assert_eq!(node.parameter("W"), Some(4), "the window is the shorter length");
    assert_eq!(node.parameter("P"), Some(3), "|6 − 4| + 1 output positions");
    // `correlate`'s result is a Float whatever the operands are
    // (`matmul.md`, `source_pattern`), so the contraction is the backend's
    // product-sum aggregate, one per output position.
    assert_eq!(
        times(node, Primitive::Float(FloatOp::ProductSum)),
        3,
        "one `product_sum_f` per output position\n{}",
        program.dump()
    );
}

// ── scalar_control: the merged array and the domain Boolean ─────────

/// `select.md`, `/array`: a merged array is a slot-wise select between two
/// bundles of the sides in one layout, at the larger capacity.
#[test]
fn a_merged_array_selects_slot_by_slot() {
    let program = run(
        vec![array_input("x", &[4]), array_input("y", &[4]), scalar_input("c")],
        vec![
            cond(
                binop("lt", load("c"), int(3)),
                vec![assign("z", load("x"))],
                vec![assign("z", load("y"))],
            ),
            ret(load("z")),
        ],
    )
    .expect("phase 1 accepts a merged array");

    let nodes = program.nodes_of(RuleId::Select);
    let node = nodes
        .iter()
        .find(|node| node.variant == Some("/array"))
        .unwrap_or_else(|| panic!("the `/array` variant\n{}", program.dump()));
    assert_eq!(node.parameter("B"), Some(4), "the capacity is the larger side's");
    assert!(times(node, Primitive::Select) >= 4, "one select per slot");
}

/// `term_to_wires.md`, the `domain` row: the domain Boolean phase 0 mints
/// for a `math.*` function whose operand carries no term is the predicate on
/// the operand's own wire.
#[test]
fn a_float_domain_condition_reaches_its_wire() {
    let program = run(
        vec![float_array_input("x", &[2])],
        vec![
            assign("a", float_scalar("x")),
            ret(call(Some("np"), "arcsin", vec![load("a")])),
        ],
    )
    .expect("phase 1 accepts a Float domain condition");

    let nodes = program.nodes_of(RuleId::TermToWires);
    let node = nodes
        .iter()
        .find(|node| node.variant == Some("domain"))
        .unwrap_or_else(|| panic!("the `domain` row\n{}", program.dump()));
    assert_eq!(
        node.fops(),
        [("gte_f", 1), ("lte_f", 1)],
        "`−1 ≤ x ≤ 1` is two comparisons\n{}",
        program.dump()
    );
    assert!(
        !program.nodes_of(RuleId::EnforceCheck).is_empty(),
        "the check has an `Enforce` root\n{}",
        program.dump()
    );
}

/// Over an array operand the same Boolean is the fold `all` of the
/// per-element condition at the valid positions of a grid batch
/// (`rules/README.md` §2, `term_to_wires.md`, the `domain` row).
#[test]
fn an_array_domain_condition_folds_every_element() {
    let program = run(
        vec![float_array_input("x", &[4])],
        vec![ret(call(Some("np"), "sqrt", vec![load("x")]))],
    )
    .expect("phase 1 accepts a domain condition over an array");

    let nodes = program.nodes_of(RuleId::TermToWires);
    let node = nodes
        .iter()
        .find(|node| node.variant == Some("domain"))
        .unwrap_or_else(|| panic!("the `domain` row\n{}", program.dump()));
    assert_eq!(
        node.fops(),
        [("gte_f", 4)],
        "`0 ≤ x` at every slot\n{}",
        program.dump()
    );
    assert_eq!(times(node, Primitive::And), 3, "the `and` fold over four slots");
    assert_eq!(times(node, Primitive::Or), 4, "an inactive slot passes");
}
