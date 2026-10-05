//! `ENUM_COORDS` (`revision/phase1/rules/reading/enum_coords.md`).
//!
//! Turns the batch of a `Fetch(v, batch, g)` into the `Coords[Q, η]` bundle:
//! per entry the r_v coordinate entries — a constant where the coordinate
//! term is answered CONST, a wire otherwise — and one active entry holding
//! `active_j = g ∧ live_j`. Every term it needs is a term requirement; the
//! rule itself emits only the thermometers of a range batch. It reads
//! nothing of v, so its node is shared by every owner and every read
//! mechanism considered for the same `Fetch`.
//!
//! **Where g enters the flags.** The rule file folds `g` into the length of
//! every thermometer, `ñ_a = ite(g, max(0, min(n_a, W_a)), 0)`, and that is
//! what the code builds: one term per thermometer axis, compiled once by
//! `TERM_TO_WIRES`, rather than one `mul` per entry to gate the product by
//! `g̃`. The gate costs Q multiplications, the term costs a `select` per
//! thermometer axis charged to the term requirement, and a batch has more
//! entries than axes, so the term is the cheaper of the two on every batch
//! the corpus has.
//!
//! **Which templates carry a thermometer.** Any template `t_a + s_a·j_a`
//! whose live end is proven: the length is phase 0's slice length
//! `range_len`, which is the grid extent for the identity template and
//! `max(0, (dims_v[a] − t_a + s_a − 1) div s_a)` for a shifted or strided
//! one.

use crate::phase0::rules::helpers::range_len;
use crate::phase0::term::TermId;

use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Batch, Capacity, Family, Interface, Live, RangeAxis};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::{Condition, Question};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct EnumCoords;

/// How the active flag of every entry is obtained (`enum_coords.md`,
/// `produced_output_bundle`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Flags {
    /// Every entry is live and `g` is constant: the constant 1.
    Always,
    /// Every entry is live: the wire of `g`, one port.
    Condition,
    /// The batch carries its own live term, one per entry: the `act(j)`
    /// port of the rule file, `Represent(g ∧ live_j, Wire[Bool])`. A
    /// consumer that reads one value into the slots of another writes such a
    /// batch — `SETITEM`'s `fetched` form reads v at ρ(p) under `p ∈ R`.
    PerEntry(Vec<crate::phase0::term::BoolId>),
    /// One thermometer per named axis, the flag of an entry the product of
    /// their bits. Each axis carries the term `ñ_a` of its length, with `g`
    /// already folded in.
    Thermometers(Vec<Thermometer>),
}

/// One thermometer axis: the axis of the batch and the term of its length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Thermometer {
    pub axis: usize,
    /// `ñ_a = ite(g, max(0, min(n_a, W_a)), 0)`, built by
    /// [`EnumCoords::length`].
    pub length: TermId,
}

impl Rule for EnumCoords {
    fn id(&self) -> RuleId {
        RuleId::EnumCoords
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Coords]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let (Some(value), Some(batch)) = (request.target.as_value(), request.batch.clone()) else {
            return Ok(Vec::new());
        };
        let value = value.clone();
        // A rank-0 read — `E-Elem`, a scalar element — has no coordinate of
        // its own: the batch is the one entry of §3.2 and the source
        // coordinates are `GATHER`'s.
        let dims = match builder.array(&value) {
            Some(array) => array.dims.clone(),
            None if builder.scalar(&value).is_some() => Vec::new(),
            None => return Ok(Vec::new()),
        };
        let mut candidate = Candidate::new(RuleId::EnumCoords, None, request.epoch);
        candidate.matched.push(value.clone());
        let under = request.condition;

        // premises: CONST per coordinate, CONST(g), PROVE(live) per axis.
        let always = builder.terms.tt();
        let constant_condition = matches!(
            builder.ask_for(
                &mut candidate,
                Question::ConstBool(request.condition),
                request.epoch,
                always,
            ),
            Verdict::ConstantBool(true)
        );
        for at in 0..batch.len() {
            super::entry_coordinates(builder, &mut candidate, &batch, at, request.epoch, under);
        }
        let Some(flags) =
            self.flags(builder, &mut candidate, &batch, &dims, request, constant_condition)
        else {
            return Ok(Vec::new());
        };

        // A coordinate that is not constant is the wire of its own term
        // requirement, and a thermometer's length is one more.
        for term in self.term_ports(builder, &batch, &flags) {
            let row = builder.child(request, term, Family::Wire, request.condition)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        // `cond`: the wire of `g`, for a batch every entry of which is live
        // under a `g` that is not CONST. A thermometer batch needs none:
        // `g` sits inside `ñ_a`.
        if matches!(flags, Flags::Condition) {
            let row = builder.child(
                request,
                Target::BoolTerm(request.condition),
                Family::Wire,
                builder.terms.tt(),
            )?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }

        // emitter: `therm(ñ_a over W_a)` per thermometer axis and one `mul`
        // per entry per undecided factor beyond the first.
        let queries = batch.len();
        if let Flags::Thermometers(axes) = &flags {
            for thermometer in axes {
                let width = self.axis_width(&batch, thermometer.axis);
                candidate = candidate.emit(Primitive::Therm { capacity: width }, 1);
            }
            let factors = axes.len() as i128 - 1;
            candidate = candidate.emit(Primitive::Mul, factors.max(0) * queries);
        }
        candidate = candidate
            .output(
                request.target.clone(),
                Interface::Coords { queries, rank: batch.rank() },
                request.condition,
            )
            .parameter("Q", queries);
        Ok(vec![candidate])
    }
}

impl EnumCoords {
    /// Which axes need a thermometer, or that every entry is live
    /// (`enum_coords.md`, premises).
    ///
    /// An axis is live everywhere when the whole template lies inside the
    /// extent; an axis whose live end alone is proven is a thermometer axis,
    /// its length phase 0's slice length. An axis live on neither end has no
    /// thermometer and the node is not offered.
    fn flags(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        batch: &Batch,
        dims: &[TermId],
        request: &RowRequest,
        constant_condition: bool,
    ) -> Option<Flags> {
        let Batch::Range { axes, live } = batch else {
            // A general batch carries its own live term; the default
            // in-range one is decided here and a supplied one becomes the
            // `act(j)` port of every entry.
            return match batch.live() {
                Live::InRange if constant_condition => Some(Flags::Always),
                Live::InRange => Some(Flags::Condition),
                Live::Term(_) => Some(self.per_entry(
                    builder,
                    batch,
                    request.condition,
                    constant_condition,
                )),
            };
        };
        if let Live::Term(_) = live {
            // A range batch's one live template is one `act` term, shared by
            // every entry of the batch.
            return Some(self.per_entry(builder, batch, request.condition, constant_condition));
        }
        let mut thermometers = Vec::new();
        for (axis, template) in axes.iter().enumerate() {
            let extent = *dims.get(axis)?;
            let base = template.base_term(&mut builder.terms);
            let (low, high) = self.ends(builder, candidate, template, base, extent, request);
            if low && high {
                continue;
            }
            // The end the entries run towards must be the open one, or the
            // axis is live on neither end and no thermometer states its
            // liveness.
            let stride = i64::try_from(template.stride).ok()?;
            let length = match (low, high, stride) {
                // Running up: the entries stop at the extent, so the count is
                // the slice length from the base to it.
                (true, false, stride) if stride > 0 => {
                    range_len(&mut builder.terms, base, extent, stride)?
                }
                // Running down: the entries stop below 0.
                (false, true, stride) if stride < 0 => {
                    let below = builder.terms.constant(-1);
                    range_len(&mut builder.terms, base, below, stride)?
                }
                _ => return None,
            };
            thermometers.push(Thermometer {
                axis,
                length: self.length(builder, length, template.width, request.condition),
            });
        }
        Some(match (thermometers.is_empty(), constant_condition) {
            (true, true) => Flags::Always,
            (true, false) => Flags::Condition,
            (false, _) => Flags::Thermometers(thermometers),
        })
    }

    /// The flags of a batch that carries its own live term: one `act(j)`
    /// port per entry, or the constant 1 where every term is literally true,
    /// which is what a region the interval layer decided leaves.
    fn per_entry(
        &self,
        builder: &mut Builder<'_>,
        batch: &Batch,
        condition: crate::phase0::term::BoolId,
        constant_condition: bool,
    ) -> Flags {
        let terms = self.active_terms(builder, batch, condition, constant_condition);
        if terms.iter().all(|&live| builder.terms.is_true(live)) {
            return if constant_condition { Flags::Always } else { Flags::Condition };
        }
        Flags::PerEntry(terms)
    }

    /// `act_j = g ∧ live_j`, the term the `act(j)` port of each entry
    /// carries (`enum_coords.md`, `required_input_ports`); `live_j` alone
    /// when CONST g = true. Duplicates fold away, because one set of
    /// conjuncts is one term id.
    fn active_terms(
        &self,
        builder: &mut Builder<'_>,
        batch: &Batch,
        condition: crate::phase0::term::BoolId,
        constant_condition: bool,
    ) -> Vec<crate::phase0::term::BoolId> {
        let entries: Vec<crate::phase0::term::BoolId> = match batch {
            Batch::General(entries) => entries
                .iter()
                .map(|entry| match entry.live {
                    Live::Term(live) => live,
                    Live::InRange => condition,
                })
                .collect(),
            _ => match batch.live() {
                Live::Term(live) => vec![live],
                Live::InRange => vec![condition],
            },
        };
        entries
            .into_iter()
            .map(|live| {
                if constant_condition {
                    live
                } else {
                    crate::phase1::build::conjunction(&mut builder.terms, &[condition, live])
                }
            })
            .collect()
    }

    /// Whether the two ends of one template are proven inside the extent:
    /// `0 ≤ t_a` and `t_a + s_a·(W_a − 1) ≤ dims_v[a] − 1`, each a built
    /// comparison (`rules/README.md` §6.3).
    fn ends(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        template: &RangeAxis,
        base: TermId,
        extent: TermId,
        request: &RowRequest,
    ) -> (bool, bool) {
        let zero = builder.literal(0);
        let offset = builder.literal(template.stride * (template.width - 1));
        let last = builder.terms.add(base, offset);
        let one = builder.literal(1);
        let top = builder.terms.sub(extent, one);
        let (lo, hi) = if template.stride >= 0 { (base, last) } else { (last, base) };
        let low = builder.prove_le(candidate, zero, lo, request.epoch, request.condition);
        let high = builder.prove_le(candidate, hi, top, request.epoch, request.condition);
        (low, high)
    }

    /// `ñ_a = ite(g, max(0, min(n_a, W_a)), 0)` (`enum_coords.md`,
    /// `required_input_ports`, `len(a)`): the length clamped into `[0, W_a]`
    /// so the thermometer always has a witness, with `g` folded in so the
    /// flags need no gate of their own.
    fn length(
        &self,
        builder: &mut Builder<'_>,
        length: TermId,
        width: Capacity,
        condition: Condition,
    ) -> TermId {
        let zero = builder.literal(0);
        let cap = builder.literal(width);
        let capped = builder.terms.min(length, cap);
        let clamped = builder.terms.max(zero, capped);
        builder.terms.ite(condition, clamped, zero)
    }

    /// The term requirements the node raises: one per coordinate leaf that
    /// is not constant, and one per thermometer length.
    ///
    /// The leaf is the template's base for a range batch, which every entry
    /// of the axis shares — `TERM_TO_WIRES` charges `t_a` once and each slot
    /// a `lin` (`enum_coords.md`, `coord(j, a)`).
    fn term_ports(
        &self,
        builder: &mut Builder<'_>,
        batch: &Batch,
        flags: &Flags,
    ) -> Vec<Target> {
        let mut out: Vec<Target> = Vec::new();
        if let Flags::PerEntry(terms) = flags {
            for &live in terms {
                if builder.terms.is_true(live) || builder.terms.is_false(live) {
                    continue;
                }
                let target = Target::BoolTerm(live);
                if !out.contains(&target) {
                    out.push(target);
                }
            }
        }
        let entries = match batch {
            Batch::Range { .. } => 1,
            _ => batch.len(),
        };
        let mut leaves: Vec<TermId> = Vec::new();
        for at in 0..entries {
            leaves.extend(batch.axis_bases(&mut builder.terms, at));
        }
        if let Flags::Thermometers(axes) = flags {
            leaves.extend(axes.iter().map(|thermometer| thermometer.length));
        }
        for term in leaves {
            if builder.terms.constant_value(term).is_some() {
                continue;
            }
            let target = Target::Term(term);
            if !out.contains(&target) {
                out.push(target);
            }
        }
        out
    }

    /// `W_a`, the width of one axis of a range batch.
    fn axis_width(&self, batch: &Batch, axis: usize) -> Capacity {
        match batch {
            Batch::Range { axes, .. } => axes.get(axis).map(|a: &RangeAxis| a.width).unwrap_or(0),
            _ => 0,
        }
    }
}
