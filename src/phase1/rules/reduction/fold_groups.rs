//! `FOLD_GROUPS` (`revision/phase1/rules/reduction/fold_groups.md`).
//!
//! Computes a reduction `w = reduce_A(v)`. Its one input is v laid out by
//! `EXPAND_FIBERS` as P fibers of W slots, a value wire and a validity bit
//! each; it folds the valid slots of every fiber with one reduction primitive
//! and outputs one wire per fiber. Where v is read is decided below it, in
//! the `EXPAND_FIBERS` node it requires, so the in-place, packed and
//! read-directly implementations of `../../../README.md` §2.3 are three nodes
//! of one requirement with one `FOLD_GROUPS` node above each.

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::interface::{
    Capacity, Family, GroupsIface, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct FoldGroups;

/// The reduction kinds `fold_groups.md`'s emitter table has a row for.
const KINDS: &[&str] = &[
    "sum", "prod", "min", "max", "amin", "amax", "any", "all", "argmax", "argmin",
    // The four that divide by the fiber's length, and so produce a Float
    // whatever the operand's dtype (`../../phase0/definition.md` §5.7). Their
    // "fiber not empty" premise is phase 0's defcond, enforced by
    // `ENFORCE_CHECK`, and is unchanged here.
    "mean", "var", "std", "norm",
];

/// The kinds whose accumulation is in the output's dtype rather than the
/// operand's, because they divide: a `mean` over an integer fiber sums in
/// Float and every slot is cast on the way in.
const DIVIDING: &[&str] = &["mean", "var", "std", "norm"];

impl Rule for FoldGroups {
    fn id(&self) -> RuleId {
        RuleId::FoldGroups
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire, Family::Store, Family::Masked, Family::Packed]
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
        let Some((operand, kind)) = self.matches(builder, &key) else {
            return Ok(Vec::new());
        };
        let Some(dtype) = super::output_dtype(builder, &key) else {
            return Ok(Vec::new());
        };
        if !super::super::sources::encoded(builder, "FOLD_GROUPS", dtype) {
            return Ok(Vec::new());
        }
        let rank = builder.array(&key).map(|array| array.rank()).unwrap_or(0);
        // A rank-0 result is one wire; a rank-1 one is a box `Store` with its
        // `Masked` and `Packed` forms over the same wires (§3.4); a rank-≥2
        // one is the `Store` alone.
        let answered = match rank {
            0 => request.family == Family::Wire,
            1 => matches!(
                request.family,
                Family::Store | Family::Masked | Family::Packed
            ),
            _ => request.family == Family::Store,
        };
        if !answered {
            return Ok(Vec::new());
        }
        let Some(operand_array) = builder.array(&operand).cloned() else {
            return Ok(Vec::new());
        };
        let Some(node) = builder.node(&key).cloned() else {
            return Ok(Vec::new());
        };
        let axes = super::axes_of(&node, operand_array.rank(), 1, "axis");

        let mut probe = Candidate::new(RuleId::FoldGroups, None, request.epoch);
        let groups = builder.groups_port(&mut probe, request, &operand, axes.clone())?;
        let under = builder.under(request.condition);
        let mut out = Vec::new();
        for entry in &groups.entries.clone() {
            let Interface::Groups(fibers) = &entry.interface else {
                continue;
            };
            // ORDER: an order-sensitive primitive needs `ordered = yes`.
            if !fibers.flags.ordered {
                continue;
            }
            let mut candidate = Candidate::new(RuleId::FoldGroups, Some(kind), request.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched = vec![key.clone(), operand.clone()];
            candidate.inputs.push(entry.requirement);
            candidate.epoch = candidate.epoch.max(entry.epoch);

            // SELECT-DROP: PROVE(d_a = W_a) per reduced axis. Proven: every
            // slot is valid and the plain step replaces the conditional one.
            let width = builder.literal(fibers.width);
            let dropped = axes.iter().all(|&axis| {
                builder.prove_eq(
                    &mut candidate,
                    operand_array.dims[axis],
                    width,
                    request.epoch,
                    under,
                )
            });
            // ZEROED: `sum`, `any` and `norm` read a zeroed `Groups` as under
            // SELECT-DROP, `enc(0)` being the identity of their step.
            let zeroed = fibers.flags.zeroed && matches!(kind, "sum" | "any");
            // SEED: a non-empty fiber lets `min`/`max` start at slot 0.
            let one = builder.literal(1);
            let seeded = fibers.flags.prefix
                && axes.iter().all(|&axis| {
                    builder.prove_le(
                        &mut candidate,
                        one,
                        operand_array.dims[axis],
                        request.epoch,
                        under,
                    )
                });

            let steps = if seeded && matches!(kind, "min" | "max" | "amin" | "amax" | "argmin" | "argmax")
            {
                fibers.width - 1
            } else {
                fibers.width
            };
            // Wrap-once (`README.md` §2): an integer sum whose unwrapped
            // value fits the field is taken in the field and wrapped once at
            // the end; otherwise every step wraps.
            let accumulator = if DIVIDING.contains(&kind) { dtype } else { fibers.dtype };
            if super::super::sources::is_float(accumulator) {
                // The Float rows: an `fop⟨name⟩` per step and the per-fiber
                // division or square root, with one `float_cast` per slot
                // when the fiber holds integers and the fold divides.
                if !super::super::sources::is_float(fibers.dtype) {
                    let passes = if matches!(kind, "var" | "std") { 2 } else { 1 };
                    candidate = candidate.emit(
                        Primitive::Float(FloatOp::FromInt),
                        passes * steps * fibers.fibers,
                    );
                }
                for (primitive, times) in float_step(kind, dropped || zeroed, seeded) {
                    candidate = candidate.emit(primitive, times * steps * fibers.fibers);
                }
                for (primitive, times) in float_finish(kind) {
                    candidate = candidate.emit(primitive, times * fibers.fibers);
                }
            } else {
                let width = builder.omega.round(builder.dtype_width(fibers.dtype));
                let once = super::wrap_once(builder, fibers.width, width);
                for (primitive, times) in
                    self.gadget(builder, kind, fibers, dropped || zeroed, seeded, once)
                {
                    candidate = candidate.emit(primitive, times * steps * fibers.fibers);
                }
                if matches!(kind, "sum") && once && width > 0 {
                    candidate = candidate.emit(
                        Primitive::Wrap {
                            width,
                            carry_width: builder.omega.round(bits_of_capacity(fibers.width) + 1),
                        },
                        fibers.fibers,
                    );
                }
            }
            candidate = self.outputs(candidate, request, &key, dtype, fibers, rank);
            candidate = candidate
                .parameter("P", fibers.fibers)
                .parameter("W", fibers.width);
            out.push(candidate);
        }
        Ok(out)
    }
}

/// One step of a Float fold, in the `fop⟨name⟩`s of `rules/README.md` §7.
///
/// A slot that may be invalid is routed through the step's identity first —
/// `select_f` for the arithmetic folds, the validity `and` for the extrema —
/// which is the same shape as the integer rows, with the float instruction
/// in place of the integer gate. `var` and `std` walk the fibers twice, as
/// `fold_groups.md` states: once for the mean and once for the sum of the
/// centred squares, so their step is the sum of the two passes.
fn float_step(kind: &str, dropped: bool, seeded: bool) -> Vec<(Primitive, i128)> {
    let f = |op: FloatOp| (Primitive::Float(op), 1i128);
    let gate = |mut rows: Vec<(Primitive, i128)>| -> Vec<(Primitive, i128)> {
        if !dropped {
            rows.insert(0, f(FloatOp::Select));
        }
        rows
    };
    match kind {
        "sum" | "mean" => gate(vec![f(FloatOp::Add)]),
        "prod" => gate(vec![f(FloatOp::Mul)]),
        // Pass one is the mean's step; pass two accumulates `(x − μ)²`.
        "var" | "std" => {
            let mut rows = gate(vec![f(FloatOp::Add)]);
            rows.extend(gate(vec![f(FloatOp::Sub), f(FloatOp::Mul), f(FloatOp::Add)]));
            rows
        }
        "norm" => gate(vec![f(FloatOp::Mul), f(FloatOp::Add)]),
        "min" | "max" | "amin" | "amax" if seeded => {
            vec![f(FloatOp::Lt), (Primitive::And, 1), f(FloatOp::Select)]
        }
        "min" | "max" | "amin" | "amax" => vec![
            f(FloatOp::Lt),
            (Primitive::And, 1),
            (Primitive::Or, 2),
            f(FloatOp::Select),
        ],
        // The extremum's index is an integer wire, so its `select` is the
        // integer one; only the comparison is a float instruction.
        "argmin" | "argmax" if seeded => {
            vec![f(FloatOp::Lt), (Primitive::And, 1), (Primitive::Select, 2)]
        }
        "argmin" | "argmax" => vec![
            f(FloatOp::Lt),
            (Primitive::And, 1),
            (Primitive::Or, 2),
            (Primitive::Select, 2),
        ],
        _ => Vec::new(),
    }
}

/// What a Float fold emits once per fiber, after the steps: the division by
/// the fiber's length and the square root.
///
/// The length is an integer wire, so it is cast before the division. `var`
/// divides twice — once for the mean its first pass accumulated, once for
/// the sum of squares — and `std` takes the root of the second.
fn float_finish(kind: &str) -> Vec<(Primitive, i128)> {
    let f = |op: FloatOp| (Primitive::Float(op), 1i128);
    match kind {
        "mean" => vec![f(FloatOp::FromInt), f(FloatOp::Div)],
        "var" => vec![f(FloatOp::FromInt), f(FloatOp::Div), f(FloatOp::Div)],
        "std" => vec![
            f(FloatOp::FromInt),
            f(FloatOp::Div),
            f(FloatOp::Div),
            f(FloatOp::Sqrt),
        ],
        "norm" => vec![f(FloatOp::Sqrt)],
        _ => Vec::new(),
    }
}

impl FoldGroups {
    /// `source_pattern`: the producer names E-Reduce-Scalar or E-Reduce-Axis,
    /// and the AST node names the kind.
    fn matches(&self, builder: &Builder<'_>, key: &ValueKey) -> Option<(ValueKey, &'static str)> {
        let entry = builder.entry(key)?;
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return None;
        };
        if !rules
            .iter()
            .any(|rule| matches!(rule, Phase0Rule::EReduceScalar | Phase0Rule::EReduceAxis))
        {
            return None;
        }
        let operand = operands
            .iter()
            .find(|operand| {
                builder.array(operand).map(|array| array.rank() > 0).unwrap_or(false)
            })?
            .clone();
        let member = builder.node(key)?.member()?;
        let kind = KINDS.iter().find(|name| **name == member)?;
        Some((operand, kind))
    }

    /// One step of the fold, in the primitives of `../README.md` §7
    /// (`fold_groups.md`, emitter).
    fn gadget(
        &self,
        builder: &Builder<'_>,
        kind: &str,
        fibers: &GroupsIface,
        dropped: bool,
        seeded: bool,
        wrap_once: bool,
    ) -> Vec<(Primitive, i128)> {
        let width = builder.omega.round(builder.dtype_width(fibers.dtype));
        let wrap = Primitive::Wrap { width, carry_width: builder.omega.round(1) };
        let comparison = Primitive::Cmp { width };
        let index_width = builder.omega.round(bits_of_capacity(fibers.width));
        match kind {
            // The wrap-once row: one `mul` per slot, or a `lin` when every
            // slot is valid; the final wrap is charged by the caller. Without
            // the policy every step wraps instead.
            "sum" if dropped && wrap_once => Vec::new(),
            "sum" if wrap_once => vec![(Primitive::Mul, 1)],
            "sum" if dropped => vec![(wrap, 1)],
            "sum" => vec![(Primitive::Mul, 1), (wrap, 1)],
            "prod" if dropped => vec![(Primitive::Mul, 1), (wrap, 1)],
            "prod" => vec![(Primitive::Mul, 1), (Primitive::Select, 1), (wrap, 1)],
            "all" if dropped => vec![(Primitive::And, 1)],
            "all" => vec![(Primitive::Mul, 1), (Primitive::And, 1)],
            "any" if dropped => vec![(Primitive::Or, 1)],
            "any" => vec![(Primitive::Or, 1), (Primitive::And, 1)],
            "min" | "max" | "amin" | "amax" if seeded => {
                vec![(comparison, 1), (Primitive::And, 1), (Primitive::Select, 1)]
            }
            "min" | "max" | "amin" | "amax" => vec![
                (comparison, 1),
                (Primitive::And, 1),
                (Primitive::Or, 2),
                (Primitive::Select, 1),
            ],
            "argmin" | "argmax" => {
                let mut rows = vec![
                    (comparison, 1),
                    (Primitive::And, 1),
                    (Primitive::Select, 2),
                ];
                if !seeded {
                    rows.push((Primitive::Or, 2));
                }
                let _ = index_width;
                rows
            }
            _ => Vec::new(),
        }
        .into_iter()
        .filter(|&(_, times)| times > 0)
        .collect()
    }

    /// The bundles of the fold's output (`fold_groups.md`,
    /// `produced_output_bundle`): one wire for a rank-0 result, a box
    /// `Store` otherwise, and the `Masked` and `Packed` forms of a rank-1
    /// result over the same wires (§3.4).
    fn outputs(
        &self,
        candidate: Candidate,
        request: &RowRequest,
        key: &ValueKey,
        dtype: Dtype,
        fibers: &GroupsIface,
        rank: usize,
    ) -> Candidate {
        if rank == 0 {
            return candidate.output(
                request.target.clone(),
                Interface::Wire(dtype),
                request.condition,
            );
        }
        let capacity: Capacity = fibers.fibers;
        let mut out = candidate.output(
            request.target.clone(),
            Interface::Store(StoreIface {
                owner: key.clone(),
                dtype,
                capacity,
                lambda: Layout::dense(LayoutKind::Box(vec![capacity]), key.clone()),
            }),
            request.condition,
        );
        if rank == 1 {
            out = out
                .output(
                    request.target.clone(),
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
                    request.target.clone(),
                    Interface::Packed {
                        dtype,
                        capacity,
                        lambda: Layout::dense(LayoutKind::Compact(capacity), key.clone()),
                        canon: false,
                    },
                    request.condition,
                );
        }
        out
    }
}
