//! `EXPORT` (`revision/phase1/rules/scalar_control/export.md`).
//!
//! Binds a delivered value to the circuit's public output. Phase 1 makes one
//! root `Represent(v, Export)` per key of R. The verifier must observe
//! elements, order, multiplicity and logical shape through a decoding fixed
//! by (P, S), so `Export` is a `Wire`, a `Packed` with its ℓ, or a dense box
//! `Store` with constant keys and its shape wires; a `Masked` or a keyed
//! `Store` is not exportable, because its holes carry prover-chosen values
//! at run-dependent positions.

use crate::phase0::desc::{Desc, Dtype};

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{ExportIface, Family, Interface, LayoutKind};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Export;

impl Rule for Export {
    fn id(&self) -> RuleId {
        RuleId::Export
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Export]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let key = match &request.target {
            Target::Value(key) => key.clone(),
            // An element of a delivered composite is named by its term
            // (`../../build.rs`, `element_target`).
            Target::Term(_) => return self.term(builder, request, Dtype::Idx),
            Target::BoolTerm(_) => return self.term(builder, request, Dtype::Bool),
        };
        let Some(entry) = builder.entry(&key) else {
            return Ok(Vec::new());
        };
        match entry.desc.clone() {
            Desc::Scalar(scalar) => {
                // A scalar's wire is declared public: one cell, no gate.
                let row = builder.child(
                    request,
                    request.target.clone(),
                    Family::Wire,
                    request.condition,
                )?;
                let mut out = Vec::new();
                for port in &row.entries {
                    let mut candidate =
                        Candidate::new(RuleId::Export, Some("scalar"), request.epoch)
                            .output(
                                request.target.clone(),
                                Interface::Export(ExportIface::Scalar(scalar.dtype)),
                                request.condition,
                            )
                            .parameter("cells", 1)
                            .emit(Primitive::Cell, 1);
                    candidate.matched.push(key.clone());
                    candidate.inputs.push(port.requirement);
                    candidate.epoch = candidate.epoch.max(port.epoch);
                    out.push(candidate);
                }
                Ok(out)
            }
            // A rank-0 array is a scalar: it has no axis, so `Export` is
            // the one cell of its wire and not a bundle (`export.md`,
            // `required_input_ports`). `np.mean(a, axis=0)` on a rank-1 `a`
            // delivers one.
            Desc::Array(array) if array.rank() == 0 => {
                self.term(builder, request, array.dtype)
            }
            Desc::Array(array) => self.array(builder, request, &array),
            // A composite in R is its elements, one root each; a value with
            // no descriptor content delivers nothing.
            _ => Ok(Vec::new()),
        }
    }
}

impl Export {
    /// A delivered term, or a rank-0 value: one public cell over the wire
    /// the `Wire` row gives, whatever rule answers it.
    fn term(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        dtype: Dtype,
    ) -> Result<Vec<Candidate>, Reject> {
        let row = builder.child(
            request,
            request.target.clone(),
            Family::Wire,
            request.condition,
        )?;
        let mut out = Vec::new();
        for port in &row.entries {
            let mut candidate = Candidate::new(RuleId::Export, Some("scalar"), request.epoch)
                .output(
                    request.target.clone(),
                    Interface::Export(ExportIface::Scalar(dtype)),
                    request.condition,
                )
                .parameter("cells", 1)
                .emit(Primitive::Cell, 1);
            candidate.inputs.push(port.requirement);
            candidate.epoch = candidate.epoch.max(port.epoch);
            out.push(candidate);
        }
        Ok(out)
    }

    /// The array forms: a `Packed` with its ℓ, or a box `Store` with its
    /// shape wires. A ranked or keyed layout is not accepted.
    fn array(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        array: &crate::phase0::desc::ArrayDesc,
    ) -> Result<Vec<Candidate>, Reject> {
        let Target::Value(key) = &request.target else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        // P1 (per axis): CONST dims_v[k] under ψ_v. A constant extent is not
        // output — the verifier knows it from S.
        let mut probe = Candidate::new(RuleId::Export, None, request.epoch);
        let mut runtime = 0i128;
        for &dimension in &array.dims {
            let under = builder.under(request.condition);
            let constant =
                builder.ask_for(&mut probe, Question::Const(dimension), request.epoch, under);
            if !matches!(constant, Verdict::Constant(_)) {
                runtime += 1;
            }
        }
        let mut out = Vec::new();
        for family in [Family::Packed, Family::Store] {
            let row = builder.child(request, Target::Value(key.clone()), family, request.condition)?;
            for port in &row.entries {
                let (capacity, canon, boxed) = match &port.interface {
                    Interface::Packed { capacity, canon, .. } => (*capacity, *canon, false),
                    Interface::Store(store) => {
                        // P2: only a box layout has constant keys and the
                        // "inside the logical shape" validity `Export` needs.
                        if !matches!(store.lambda.layout, LayoutKind::Box(_)) {
                            continue;
                        }
                        (store.capacity, true, true)
                    }
                    _ => continue,
                };
                let mut candidate = Candidate::new(RuleId::Export, Some("array"), request.epoch)
                    .output(
                        request.target.clone(),
                        Interface::Export(ExportIface::Array {
                            dtype: array.dtype,
                            capacity,
                        }),
                        request.condition,
                    )
                    .parameter("B", capacity)
                    .parameter("cells", runtime + capacity)
                    .emit(Primitive::Cell, runtime + capacity);
                candidate.matched.push(key.clone());
                candidate.inputs.push(port.requirement);
                candidate.epoch = candidate.epoch.max(port.epoch);
                candidate.answers = probe.answers.clone();
                // emitter: the canonical tail, absent when the bundle is
                // already canonical or every axis is CONST (P3,
                // `Ω.export_canonical`). Its thermometer is the
                // `Ranks(I, valid)` node of the bundle (§4.1), a port.
                if builder.omega.export_canonical && runtime > 0 && !canon {
                    let Some(subject) =
                        crate::phase1::rules::masks::subject_of(&port.interface, &key, true)
                    else {
                        continue;
                    };
                    let row = builder.ranks_port(&mut candidate, request, &key, subject)?;
                    let Some(first) = row.entries.first() else {
                        continue;
                    };
                    candidate.inputs.push(first.requirement);
                    candidate.epoch = candidate.epoch.max(first.epoch);
                    candidate = candidate
                        .emit(Primitive::Mul, capacity)
                        .parameter("w", bits_of_capacity(capacity) as i128);
                }
                if boxed && builder.omega.export_canonical && runtime > 0 {
                    candidate = candidate.emit(Primitive::Mul, capacity);
                }
                out.push(candidate);
            }
        }
        Ok(out)
    }
}
