//! `SORT` (`revision/phase1/rules/ordering/sort.md`).
//!
//! `v = sort(a, axis=k)`: the prover supplies the sorted list as witness,
//! one `perm` checks that it is a rearrangement of the operand's valid
//! elements — the tuples carry the validity bit, so the valid results are
//! exactly the valid operand elements — and `B − 1` gated adjacent range
//! checks check that it is non-decreasing. A non-decreasing sequence with a
//! given multiset is unique, so the witness is forced.
//!
//! A sort keeps its operand's length: the rule asks no MAX, inherits the
//! capacity from the operand's bundle, and the result's length wire is the
//! operand's.
//!
//! Variant `dyn` reads a rank-1 operand through one Store-family bundle and
//! answers a `Packed`, a `Masked` and a box `Store` from one node (§3.4);
//! `dyn.canon` zeroes the `Packed`'s tail; `grid` sorts an operand of
//! constant shape fiber by fiber and answers a box `Store`.

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, Capacity, Family, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Sort;

/// What the source pattern read: the operand and the normalised axis.
pub(crate) struct Matched {
    pub operand: ValueKey,
    pub axis: usize,
    /// Whether the call is the `argsort` form.
    pub indices: bool,
}

impl Rule for Sort {
    fn id(&self) -> RuleId {
        RuleId::Sort
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Packed, Family::Masked, Family::Store]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(key) = request.target.as_value() else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        let Some(matched) = matches(builder, &key) else {
            return Ok(Vec::new());
        };
        if matched.indices {
            return Ok(Vec::new());
        }
        build_sort(builder, request, &key, &matched, RuleId::Sort, None)
    }
}

/// `source_pattern`: `Applied([…, E-Sort], [a], _)`, with `axis` and `kind`
/// read off the AST node; `kind` is read and ignored (README §2.5), and an
/// `order=` argument makes the node not match.
pub(crate) fn matches(builder: &Builder<'_>, key: &ValueKey) -> Option<Matched> {
    let entry = builder.entry(key)?;
    let Producer::Applied { rules, operands, .. } = &entry.producer else {
        return None;
    };
    // A value that was sorted and then reshaped is one key with two rules,
    // and the sorted value itself has no key of its own.
    if !rules.iter().all(|rule| *rule == Phase0Rule::ESort) || rules.is_empty() {
        return None;
    }
    let array = entry.desc.as_array()?;
    let operand = operands
        .iter()
        .find(|operand| builder.array(operand).is_some())?
        .clone();
    let source = builder.array(&operand)?;
    if !super::super::sources::has_encoding(source.dtype) {
        return None;
    }
    let node = builder.node(key);
    if node.and_then(|node| node.keyword("order")).is_some() {
        return None;
    }
    // The `argsort` form is told from the `sort` form by the call, and by
    // the result dtype where the call is not on the node.
    let member = node.and_then(|node| node.member()).unwrap_or("");
    let indices = member.contains("argsort") || (member.is_empty() && array.dtype == Dtype::Idx);
    let rank = source.rank();
    let axis = node
        .and_then(|node| node.parameter(1, "axis"))
        .and_then(|value| value.int())
        .map(|value| if value < 0 { value + rank as i64 } else { value })
        .unwrap_or(rank as i64 - 1)
        .max(0) as usize;
    (axis < rank).then_some(Matched { operand, axis, indices })
}

/// The nodes of `SORT` and of `ARGSORT`, which differ in three numbers: the
/// permutation carries the index column, the order check is strict on the
/// combined `(value, index)` key and so is `b` bits wider, and the index
/// vector is one witness more (`argsort.md`, `emitter`). `also` names the
/// value whose sorted elements are a second bundle of the same wires.
pub(crate) fn build_sort(
    builder: &mut Builder<'_>,
    request: &RowRequest,
    key: &ValueKey,
    matched: &Matched,
    rule: RuleId,
    also: Option<&ValueKey>,
) -> Result<Vec<Candidate>, Reject> {
    let Some(array) = builder.array(key).cloned() else {
        return Ok(Vec::new());
    };
    if array.rank() == 0 || matched.axis >= array.rank() {
        return Ok(Vec::new());
    }
    // β is the *operand*'s: `ARGSORT`'s own dtype is Idx, and what the
    // order check compares is the values the permutation carried.
    let Some(operand_dtype) = builder.array(&matched.operand).map(|a| a.dtype) else {
        return Ok(Vec::new());
    };
    let Some(order) = super::order_key(builder, operand_dtype) else {
        return Ok(Vec::new());
    };
    let width = super::ord_width(order);
    let mut out = Vec::new();
    let under = builder.under(request.condition);
    // premises: CONST dims_a[0] under g selects the constant fold; neither
    // answer drops the rule.
    let mut probe = Candidate::new(rule, None, request.epoch);
    let constant = match builder.ask_for(
        &mut probe,
        Question::Const(array.dims[matched.axis]),
        request.epoch,
        under,
    ) {
        Verdict::Constant(value) => Some(value.max(0)),
        _ => None,
    };

    if array.rank() == 1 {
        for family in [Family::Store, Family::Masked, Family::Packed] {
            let psi = builder.psi(&matched.operand);
            let condition =
                builder.child_under_g(&mut probe, request, psi, &matched.operand.encode())?;
            let row = builder.child(
                request,
                Target::Value(matched.operand.clone()),
                family,
                condition,
            )?;
            for entry in &row.entries.clone() {
                let Some(operand) = super::operand_of(&entry.interface) else {
                    continue;
                };
                let mut candidate = Candidate::new(rule, Some("dyn"), request.epoch);
                candidate.answers = probe.answers.clone();
                candidate.matched.push(key.clone());
                candidate.inputs.push(entry.requirement);
                candidate.epoch = candidate.epoch.max(entry.epoch);
                let capacity = operand.capacity;
                // emitter: the prefix validity of the result, the permutation
                // over the (value, validity) pairs, and the order check.
                let dense = operand.dense && constant.is_some();
                if operand.thermometer && constant.is_none() {
                    candidate = candidate.emit(Primitive::Therm { capacity }, 1);
                }
                let pairs = constant.unwrap_or(capacity) - 1;
                // `ARGSORT` carries the logical index through the
                // permutation and orders on the combined key.
                let indices = u32::from(matched.indices);
                let index_width = indices * bits_of_capacity(capacity);
                candidate = candidate
                    .emit(
                        Primitive::Perm {
                            capacity: if dense { constant.unwrap_or(capacity) } else { capacity },
                            arity: if dense { 1 } else { 2 } + indices,
                        },
                        1,
                    )
                    .emit(Primitive::Witness, (2 + indices as i128) * capacity);
                // `ARGSORT` orders strictly on the combined key, `SORT` on
                // the value alone and not strictly.
                candidate = super::order_check(
                    candidate,
                    order,
                    pairs,
                    index_width,
                    matched.indices,
                    constant.is_none(),
                );
                candidate = candidate
                    .parameter("B", capacity)
                    .parameter("beta", width as i128)
                    .parameter("b", index_width as i128);
                for target in std::iter::once(key).chain(also) {
                    candidate = bundles(candidate, builder, request, target, &array, capacity);
                }
                out.push(candidate.clone());
                // `dyn.canon`: the `Packed`'s tail zeroed, at B more `mul`.
                let mut canonical = candidate;
                canonical.variant = Some("dyn.canon");
                canonical = canonical.emit(Primitive::Mul, capacity);
                for target in std::iter::once(key).chain(also) {
                    canonical = canonical.output(
                        Target::Value(target.clone()),
                        Interface::Packed {
                            dtype: builder.array(target).map(|a| a.dtype).unwrap_or(array.dtype),
                            capacity,
                            lambda: Layout::dense(
                                LayoutKind::Compact(capacity),
                                target.clone(),
                            ),
                            canon: true,
                        },
                        request.condition,
                    );
                }
                out.push(canonical);
            }
        }
        return Ok(out);
    }

    // `grid`: an operand of constant shape, sorted fiber by fiber.
    let mut extents = Vec::with_capacity(array.dims.len());
    for &dimension in &array.dims {
        match builder.ask_for(&mut probe, Question::Const(dimension), request.epoch, under) {
            Verdict::Constant(value) => extents.push(value.max(0)),
            _ => return Ok(out),
        }
    }
    let mut candidate = Candidate::new(rule, Some("grid"), request.epoch);
    candidate.answers = probe.answers.clone();
    candidate.matched.push(key.clone());
    let batch = Batch::grid(&extents);
    let values = builder.fetch_port(&mut candidate, request, &matched.operand, &batch)?;
    let Some(first) = values.entries.first() else {
        return Ok(out);
    };
    candidate.inputs.push(first.requirement);
    candidate.epoch = candidate.epoch.max(first.epoch);
    let slots: Capacity = extents.iter().product();
    let along = extents[matched.axis];
    let fibers = if along > 0 { slots / along } else { 0 };
    let indices = u32::from(matched.indices);
    let index_width = indices * bits_of_capacity(along);
    candidate = candidate
        .emit(Primitive::Perm { capacity: along, arity: 1 + indices }, fibers)
        .emit(Primitive::Witness, (1 + indices as i128) * slots);
    candidate = super::order_check(
        candidate,
        order,
        fibers * (along - 1).max(0),
        index_width,
        matched.indices,
        false,
    );
    candidate = candidate
        .parameter("Q", slots)
        .parameter("F", fibers)
        .parameter("beta", width as i128)
        .parameter("b", index_width as i128);
    for target in std::iter::once(key).chain(also) {
        let dtype = builder.array(target).map(|a| a.dtype).unwrap_or(array.dtype);
        candidate = candidate.output(
            Target::Value(target.clone()),
            Interface::Store(StoreIface {
                owner: target.clone(),
                dtype,
                capacity: slots,
                lambda: Layout::dense(LayoutKind::Box(extents.clone()), target.clone()),
            }),
            request.condition,
        );
    }
    out.push(candidate);
    Ok(out)
}

/// The three bundles of one set of `dyn` wires (§3.4): the `Packed` the
/// consumers scan, the `Masked` whose validity is the prefix, and the box
/// `Store` with the constant keys.
fn bundles(
    candidate: Candidate,
    builder: &Builder<'_>,
    request: &RowRequest,
    key: &ValueKey,
    array: &crate::phase0::desc::ArrayDesc,
    capacity: Capacity,
) -> Candidate {
    let dtype = builder.array(key).map(|a| a.dtype).unwrap_or(array.dtype);
    candidate
        .output(
            Target::Value(key.clone()),
            Interface::Packed {
                dtype,
                capacity,
                lambda: Layout::dense(LayoutKind::Compact(capacity), key.clone()),
                canon: false,
            },
            request.condition,
        )
        .output(
            Target::Value(key.clone()),
            Interface::Masked {
                dtype,
                capacity,
                lambda: Layout::dense(
                    LayoutKind::Ranked { capacity, ranks: None },
                    key.clone(),
                ),
            },
            request.condition,
        )
        .output(
            Target::Value(key.clone()),
            Interface::Store(StoreIface {
                owner: key.clone(),
                dtype,
                capacity,
                lambda: Layout::dense(LayoutKind::Box(vec![capacity]), key.clone()),
            }),
            request.condition,
        )
        .parameter("w", bits_of_capacity(capacity) as i128)
}
