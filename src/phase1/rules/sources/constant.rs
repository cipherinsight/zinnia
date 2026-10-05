//! `CONSTANT` (`revision/phase1/rules/sources/constant.md`).
//!
//! The array values whose every element is a compile-time function of its
//! coordinate: literal arrays and the shape-only constructors whose extents
//! are literals or terms of origin shape. An element is a literal held in
//! the bundle without a wire. When every extent is a literal the node has no
//! wire and no constraint (`table`). When an extent is a term the elements
//! are still literals per slot but the extent lives on a wire: the node
//! takes it through a term requirement, the capacity of the axis is a MAX
//! question, and the validity bits are a thermometer (`box`, `packed`).

use crate::phase0::desc::Dtype;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::{ReqId, Target};
use crate::phase1::interface::{Capacity, Family, Interface, Layout, LayoutKind, StoreIface};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Constant;

/// The constructors of the rule file's `source_pattern` table whose elements
/// are literals per slot.
const CONSTRUCTORS: &[&str] = &[
    "zeros", "ones", "empty", "zeros_like", "ones_like", "empty_like", "full", "full_like",
    "identity", "eye", "arange", "linspace", "indices", "array", "asarray",
];

impl Rule for Constant {
    fn id(&self) -> RuleId {
        RuleId::Constant
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Store, Family::Masked, Family::Packed]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Target::Value(key) = &request.target else {
            return Ok(Vec::new());
        };
        if builder.entry(key).is_none() {
            return Ok(Vec::new());
        }
        // `empty`: a value with no elements at all, whatever produced it.
        let empty = self.empty(builder, request)?;
        if !empty.is_empty() {
            return Ok(empty);
        }
        let Some(entry) = builder.entry(key) else {
            return Ok(Vec::new());
        };
        // source_pattern: the producer names E-Zeros or E-Arange, and the
        // AST node is one of the constructors of the table.
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return Ok(Vec::new());
        };
        if !rules.iter().any(|rule| {
            matches!(rule, Phase0Rule::EZeros | Phase0Rule::EOnes | Phase0Rule::EArange)
        }) {
            return Ok(Vec::new());
        }
        let member = builder
            .node(key)
            .and_then(|node| node.member())
            .unwrap_or_default()
            .to_string();
        if !CONSTRUCTORS.contains(&member.as_str()) {
            return Ok(Vec::new());
        }
        let Some(array) = entry.desc.as_array() else {
            return Ok(Vec::new());
        };
        if !super::has_encoding(array.dtype) {
            return Ok(Vec::new());
        }
        let (dtype, dims) = (array.dtype, array.dims.clone());
        let operands = operands.clone();

        // premises: CONST dims_v[k] under g per axis, and CONST g itself.
        let mut probe = Candidate::new(RuleId::Constant, None, request.epoch);
        let mut capacities: Vec<Capacity> = Vec::with_capacity(dims.len());
        let mut runtime: Vec<usize> = Vec::new();
        let under = builder.under(request.condition);
        for (axis, &dimension) in dims.iter().enumerate() {
            match builder.ask_for(
                &mut probe,
                Question::Const(dimension),
                request.epoch,
                under,
            ) {
                Verdict::Constant(extent) => capacities.push(extent.max(0)),
                _ => {
                    // resource_relations: the capacity of a term axis is the
                    // MAX of its dimension term over Φ_e ∧ g.
                    let bound = builder.ask_for(
                        &mut probe,
                        Question::Max(dimension),
                        request.epoch,
                        under,
                    );
                    let Verdict::Bound(capacity) = bound else {
                        return Ok(Vec::new());
                    };
                    capacities.push(capacity.max(0));
                    runtime.push(axis);
                }
            }
        }
        let gated = !builder.terms.is_true(request.condition);
        let capacity: Capacity = capacities.iter().product();

        let mut ports: Vec<(ReqId, u32)> = Vec::new();
        for &axis in &runtime {
            let row = builder.child(
                request,
                Target::Term(dims[axis]),
                Family::Wire,
                request.condition,
            )?;
            let Some(first) = row.entries.first() else {
                // No wire for the extent, so no bundle: the report names the
                // axis rather than offering a node without its port.
                return Ok(Vec::new());
            };
            ports.push((first.requirement, first.epoch));
        }
        if gated {
            let row = builder.child(
                request,
                Target::BoolTerm(request.condition),
                Family::Wire,
                builder.terms.tt(),
            )?;
            if let Some(first) = row.entries.first() {
                ports.push((first.requirement, first.epoch));
            }
        }
        // The `fill` port of `full` with a scalar value: the element at every
        // slot is that wire.
        let fill = self.fill_port(builder, &mut probe, request, &member, &operands)?;
        if let Some(port) = fill {
            ports.push(port);
        }

        let epoch = ports
            .iter()
            .map(|&(_, epoch)| epoch)
            .fold(request.epoch, u32::max);
        let mut out = Vec::new();
        out.push(self.bundle(
            request,
            dtype,
            &capacities,
            &runtime,
            &ports,
            epoch,
            gated,
            fill.is_some(),
            &probe,
        ));
        if !runtime.is_empty() && runtime.iter().all(|&axis| axis == 0) {
            let width = builder.omega.round(bits_of_capacity(capacities[0]));
            let mut candidate = Candidate::new(RuleId::Constant, Some("packed"), epoch)
                .output(
                    request.target.clone(),
                    Interface::Packed {
                        dtype,
                        capacity,
                        lambda: Layout::dense(LayoutKind::Compact(capacity), key.clone()),
                        canon: false,
                    },
                    request.condition,
                )
                .parameter("B", capacity)
                .parameter("w", width as i128)
                .emit(Primitive::Range { width }, 2);
            if gated {
                candidate = candidate.emit(Primitive::Mul, 1);
            }
            candidate.matched.push(key.clone());
            candidate.inputs = ports.iter().map(|&(at, _)| at).collect();
            candidate.answers = probe.answers.clone();
            out.push(candidate);
        }
        Ok(out)
    }
}

impl Constant {
    /// `empty`: an array value with no element at all
    /// (`sources/constant.md`, the `empty` variant).
    ///
    /// `np.array([])` and `np.array([[], [], []])` reach phase 1 as an
    /// `E-Stack` of an empty list behind an `E-Copy`, and every other value
    /// whose shape has a zero axis is the same thing: the bundle has no
    /// slot, so there is nothing to read, nothing to constrain and no port to
    /// fill, and every fold over it is its identity because no slot is
    /// valid. The node is offered wherever every extent is CONST and their
    /// product is 0, so the shape and not the producer decides it; an input
    /// is left to `INPUT_STORE` and an alias to the value it names.
    fn empty(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Target::Value(key) = &request.target else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        let Some(entry) = builder.entry(&key) else {
            return Ok(Vec::new());
        };
        if matches!(entry.producer, Producer::Input) || entry.producer.relabels().is_some() {
            return Ok(Vec::new());
        }
        let Some(array) = entry.desc.as_array() else {
            return Ok(Vec::new());
        };
        if array.rank() == 0 || !super::has_encoding(array.dtype) {
            return Ok(Vec::new());
        }
        let (dtype, dims) = (array.dtype, array.dims.clone());
        let mut candidate = Candidate::new(RuleId::Constant, Some("empty"), request.epoch);
        let under = builder.under(request.condition);
        let mut axes: Vec<Capacity> = Vec::with_capacity(dims.len());
        for &dimension in &dims {
            let Verdict::Constant(extent) =
                builder.ask_for(&mut candidate, Question::Const(dimension), request.epoch, under)
            else {
                return Ok(Vec::new());
            };
            axes.push(extent.max(0));
        }
        if axes.iter().product::<Capacity>() != 0 {
            return Ok(Vec::new());
        }
        candidate.matched.push(key.clone());
        candidate = candidate
            .output(
                request.target.clone(),
                Interface::Store(StoreIface {
                    owner: key.clone(),
                    dtype,
                    capacity: 0,
                    lambda: Layout::dense(LayoutKind::Box(axes.clone()), key.clone()),
                }),
                request.condition,
            )
            .output(
                request.target.clone(),
                Interface::Packed {
                    dtype,
                    capacity: 0,
                    lambda: Layout::dense(LayoutKind::Compact(0), key.clone()),
                    canon: true,
                },
                request.condition,
            )
            .parameter("B", 0);
        for (axis, &capacity) in axes.iter().enumerate() {
            candidate = candidate.parameter(&format!("B_{}", axis), capacity);
        }
        Ok(vec![candidate])
    }

    /// The `table` and `box` variants, which answer `Store`, `Masked` (rank
    /// 1) and `Packed(canon = yes)` (T ⊆ {0}) from one node (§3.4).
    #[allow(clippy::too_many_arguments)]
    fn bundle(
        &self,
        request: &RowRequest,
        dtype: Dtype,
        capacities: &[Capacity],
        runtime: &[usize],
        ports: &[(ReqId, u32)],
        epoch: u32,
        gated: bool,
        fill: bool,
        probe: &Candidate,
    ) -> Candidate {
        let Target::Value(key) = &request.target else {
            unreachable!("the caller matched a value");
        };
        let capacity: Capacity = capacities.iter().product();
        let variant = if runtime.is_empty() { "table" } else { "box" };
        let mut candidate = Candidate::new(RuleId::Constant, Some(variant), epoch)
            .output(
                request.target.clone(),
                Interface::Store(StoreIface {
                    owner: key.clone(),
                    dtype,
                    capacity,
                    lambda: Layout::dense(LayoutKind::Box(capacities.to_vec()), key.clone()),
                }),
                request.condition,
            )
            .parameter("B", capacity);
        candidate.matched.push(key.clone());
        candidate.inputs = ports.iter().map(|&(at, _)| at).collect();
        candidate.answers = probe.answers.clone();
        if capacities.len() == 1 {
            candidate = candidate.output(
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
            );
        }
        if runtime.iter().all(|&axis| axis == 0) {
            let lambda = Layout::dense(LayoutKind::Compact(capacity), key.clone());
            candidate = candidate.output(
                request.target.clone(),
                Interface::Packed { dtype, capacity, lambda: lambda.clone(), canon: true },
                request.condition,
            );
            if runtime.is_empty() {
                // `table`: the tail is empty, so both canon forms are the
                // same wires.
                candidate = candidate.output(
                    request.target.clone(),
                    Interface::Packed { dtype, capacity, lambda, canon: false },
                    request.condition,
                );
            }
        }
        // emitter: the gate, one thermometer per term axis, the product when
        // two or more vary, and the zeroed tail when the fill is a wire.
        if gated {
            candidate = candidate.emit(Primitive::Mul, runtime.len() as i128);
        }
        for &axis in runtime {
            candidate = candidate.emit(Primitive::Therm { capacity: capacities[axis] }, 1);
        }
        if runtime.len() >= 2 {
            candidate = candidate.emit(Primitive::Mul, (runtime.len() as i128 - 1) * capacity);
        }
        if fill && !runtime.is_empty() {
            candidate = candidate.emit(Primitive::Mul, capacity);
        }
        candidate
    }

    /// `full(shape, s)` with a scalar value holds the wire of `s` at every
    /// slot, by alias; that wire is a port.
    fn fill_port(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        request: &RowRequest,
        member: &str,
        operands: &[crate::phase0::desc::ValueKey],
    ) -> Result<Option<(ReqId, u32)>, Reject> {
        if !matches!(member, "full" | "full_like") {
            return Ok(None);
        }
        let Some(fill) = operands.last() else {
            return Ok(None);
        };
        let Some(scalar) = builder.scalar(fill) else {
            return Ok(None);
        };
        // A literal fill is folded into the consumer's gate, so it is no
        // port; a scalar with no term has no wire the library states.
        match scalar.term {
            crate::phase0::desc::ScalarTerm::Int(t) => {
                if builder.terms.constant_value(t).is_some() {
                    return Ok(None);
                }
            }
            crate::phase0::desc::ScalarTerm::Bool(_) => {}
            crate::phase0::desc::ScalarTerm::Opaque => return Ok(None),
        }
        let psi = builder.psi(fill);
        let condition = builder.child_under_g(candidate, request, psi, &fill.encode())?;
        let row = builder.child(request, Target::Value(fill.clone()), Family::Wire, condition)?;
        Ok(row
            .entries
            .first()
            .map(|entry| (entry.requirement, entry.epoch)))
    }
}
