//! `INPUT_STORE` (`revision/phase1/rules/sources/input_store.md`).
//!
//! Binds an array parameter to the bundles of the library without moving a
//! value. The front end fixes how the parameter arrives: per axis k a
//! capacity N_k ∈ S, a payload of B = ∏_k N_k cells in C-order, one extent
//! cell per variable-length axis. The `bind` variant binds those cells and
//! enforces the admitted domain on them; `box` derives the validity bits a
//! `Store` or a `Masked` needs; `packed` and `packed_canon` derive a
//! `Packed`. The capacities are the N_k of S and never a MAX question,
//! because the encoding has N_k cells whatever Φ says.

use crate::phase0::phi::VarOrigin;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Capacity, Family, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct InputStore;

/// What the source pattern read off the parameter.
struct Match {
    key: crate::phase0::desc::ValueKey,
    dtype: crate::phase0::desc::Dtype,
    /// Per axis: the capacity N_k, and the dimension term when the axis is
    /// variable-length.
    axes: Vec<(Capacity, Option<TermId>)>,
    private: bool,
    hashed: bool,
}

impl Match {
    fn capacity(&self) -> Capacity {
        self.axes.iter().map(|(n, _)| *n).product()
    }

    fn grid(&self) -> Vec<Capacity> {
        self.axes.iter().map(|(n, _)| *n).collect()
    }

    /// K of the rule file: the axes whose extent is a variable.
    fn runtime_axes(&self) -> Vec<usize> {
        self.axes
            .iter()
            .enumerate()
            .filter_map(|(at, (_, term))| term.map(|_| at))
            .collect()
    }
}

impl Rule for InputStore {
    fn id(&self) -> RuleId {
        RuleId::InputStore
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Payload, Family::Store, Family::Masked, Family::Packed]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(matched) = self.matches(builder, request) else {
            return Ok(Vec::new());
        };
        match request.family {
            Family::Payload => self.bind(builder, request, &matched),
            Family::Store | Family::Masked => self.boxed(builder, request, &matched),
            Family::Packed => self.packed(builder, request, &matched),
            _ => Ok(Vec::new()),
        }
    }
}

impl InputStore {
    fn matches(&self, builder: &Builder<'_>, request: &RowRequest) -> Option<Match> {
        let Target::Value(key) = &request.target else {
            return None;
        };
        let entry = builder.entry(key)?;
        if !matches!(entry.producer, Producer::Input) {
            return None;
        }
        let array = entry.desc.as_array()?;
        if array.rank() == 0 || !super::has_encoding(array.dtype) {
            return None;
        }
        let parameter = builder.input_node(key)?;
        let mut axes = Vec::with_capacity(array.rank());
        for &dimension in &array.dims {
            match builder.terms.constant_value(dimension) {
                Some(extent) => axes.push((extent, None)),
                // premises: every variable axis has a finite N_k, which is
                // the declared maximum phase 0 gave the variable.
                None => axes.push((builder.interval(dimension).hi?, Some(dimension))),
            }
        }
        let hashed = super::declared(&parameter.annotation)
            .map(|(_, hashed)| hashed)
            .unwrap_or(false);
        // premises: a hashed input has K = ∅ (the group's Open list).
        if hashed && axes.iter().any(|(_, term)| term.is_some()) {
            return None;
        }
        Some(Match {
            key: key.clone(),
            dtype: array.dtype,
            axes,
            private: super::is_private(&parameter.annotation),
            hashed,
        })
    }

    /// `bind`: the encoding cells, domain-checked (B1 to B4).
    fn bind(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        matched: &Match,
    ) -> Result<Vec<Candidate>, Reject> {
        let capacity = matched.capacity();
        let width = builder.dtype_width(matched.dtype);
        let mut candidate = Candidate::new(RuleId::InputStore, Some("bind"), request.epoch)
            .output(
                request.target.clone(),
                Interface::Payload {
                    dtype: matched.dtype,
                    capacity,
                    grid: matched.grid(),
                },
                request.condition,
            )
            .parameter("B", capacity);
        candidate.matched.push(matched.key.clone());

        // (B1) 0 ≤ ext[k] ≤ N_k, two range checks per variable axis.
        for &axis in &matched.runtime_axes() {
            let extent_width = builder.omega.round(bits_of_capacity(matched.axes[axis].0));
            candidate = candidate.emit(Primitive::Range { width: extent_width }, 2);
            candidate = candidate.parameter(&format!("w_{}", axis), extent_width as i128);
        }
        // (B2) the dtype range on every slot, holes included. A Float slot
        // has none: its well-formedness is the backend's encoding
        // (`rules/README.md` §5), so the library states no gadget for it.
        if matched.private || builder.omega.range_check_public {
            if matched.dtype == crate::phase0::desc::Dtype::Bool {
                candidate = candidate.emit(Primitive::Bool, capacity);
            } else if !super::is_float(matched.dtype) {
                candidate = candidate.emit(Primitive::Range { width }, capacity);
            }
        }
        // (B3) an axis whose symbol another parameter introduced is equated
        // to that parameter's extent wire: one `lin`, and one port.
        //
        // Which parameter that is: the one `VarOrigin::InputLength` names,
        // and for a `SymbolicNDArray` symbol shared by name, which records
        // no parameter, the first parameter in Δ order that binds it
        // (`rules/README.md` §2), found by scanning the inputs' dims.
        for &axis in &matched.runtime_axes() {
            let Some(term) = matched.axes[axis].1 else {
                continue;
            };
            let Some(variable) = builder.terms.var_of(term) else {
                continue;
            };
            let owner = match builder.var_origin(variable).clone() {
                VarOrigin::InputLength { input, .. } => builder.input_key(&input).cloned(),
                VarOrigin::ShapeSymbol { .. } => builder.first_input_with_dim(term),
                _ => continue,
            };
            let Some(owner) = owner else {
                continue;
            };
            if owner == matched.key {
                continue;
            }
            let row = builder.child(
                request,
                Target::Term(term),
                Family::Wire,
                builder.terms.tt(),
            )?;
            let Some(first) = row.entries.first() else {
                continue;
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        // (B4) the hash over the payload cells.
        if matched.hashed {
            candidate = candidate.emit(Primitive::Hash { cells: capacity }, 1);
        }
        Ok(vec![candidate])
    }

    /// `box`: a thermometer per variable axis; `Store` and, for rank 1,
    /// `Masked`, from one node (§3.4).
    fn boxed(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        matched: &Match,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(payload) = self.payload_port(builder, request, matched)? else {
            return Ok(Vec::new());
        };
        let capacity = matched.capacity();
        let runtime = matched.runtime_axes();
        let lambda = Layout::dense(LayoutKind::Box(matched.grid()), matched.key.clone());
        let mut candidate = Candidate::new(RuleId::InputStore, Some("box"), request.epoch)
            .output(
                request.target.clone(),
                Interface::Store(StoreIface {
                    owner: matched.key.clone(),
                    dtype: matched.dtype,
                    capacity,
                    lambda: lambda.clone(),
                }),
                request.condition,
            )
            .parameter("B", capacity);
        candidate.matched.push(matched.key.clone());
        candidate.inputs.push(payload.0);
        candidate.epoch = candidate.epoch.max(payload.1);
        if matched.axes.len() == 1 {
            candidate = candidate.output(
                request.target.clone(),
                Interface::Masked {
                    dtype: matched.dtype,
                    capacity,
                    lambda: Layout::dense(
                        LayoutKind::Ranked { capacity, ranks: None },
                        matched.key.clone(),
                    ),
                },
                request.condition,
            );
        }
        // (X1) one thermometer per variable axis, (X2) the product of the
        // per-axis bits when two or more axes vary.
        for &axis in &runtime {
            candidate = candidate.emit(
                Primitive::Therm { capacity: matched.axes[axis].0 },
                1,
            );
        }
        if runtime.len() >= 2 {
            candidate = candidate.emit(Primitive::Mul, (runtime.len() as i128 - 1) * capacity);
        }
        Ok(vec![candidate])
    }

    /// `packed` and `packed_canon`, admitted when K ⊆ {0}.
    fn packed(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        matched: &Match,
    ) -> Result<Vec<Candidate>, Reject> {
        let runtime = matched.runtime_axes();
        if runtime.iter().any(|&axis| axis != 0) {
            return Ok(Vec::new());
        }
        let Some(payload) = self.payload_port(builder, request, matched)? else {
            return Ok(Vec::new());
        };
        let capacity = matched.capacity();
        let lambda = Layout::dense(LayoutKind::Compact(capacity), matched.key.clone());
        let mut out = Vec::with_capacity(2);
        for canon in [false, true] {
            let mut candidate = Candidate::new(
                RuleId::InputStore,
                Some(if canon { "packed_canon" } else { "packed" }),
                request.epoch,
            )
            .output(
                request.target.clone(),
                Interface::Packed {
                    dtype: matched.dtype,
                    capacity,
                    lambda: lambda.clone(),
                    canon,
                },
                request.condition,
            )
            .parameter("B", capacity);
            candidate.matched.push(matched.key.clone());
            candidate.inputs.push(payload.0);
            candidate.epoch = candidate.epoch.max(payload.1);
            // The tail is empty when every axis is fixed, so the canonical
            // form costs nothing there.
            if canon && !runtime.is_empty() {
                candidate = candidate
                    .emit(Primitive::Therm { capacity: matched.axes[0].0 }, 1)
                    .emit(Primitive::Mul, capacity);
            }
            out.push(candidate);
        }
        Ok(out)
    }

    /// The `enc` port every derived variant requires: `Represent(x,
    /// Payload(x))`, so phase 2 counts the domain checks once whichever
    /// bundles of x are selected.
    fn payload_port(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        matched: &Match,
    ) -> Result<Option<(crate::phase1::graph::ReqId, u32)>, Reject> {
        let row = builder.child(
            request,
            Target::Value(matched.key.clone()),
            Family::Payload,
            builder.terms.tt(),
        )?;
        Ok(row
            .entries
            .first()
            .map(|entry| (entry.requirement, entry.epoch)))
    }
}
