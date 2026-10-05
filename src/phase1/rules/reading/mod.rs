//! Rule group `reading` (`revision/phase1/rules/reading/README.md`).
//!
//! The group answers `Fetch(v, batch, g)`: the elements of v at the entries
//! of a batch, delivered as `Values[τ, Q, η]`. It holds the two steps before
//! a read — turning the batch into coordinates and active flags
//! (`ENUM_COORDS`), and applying the map from v to the owner that holds its
//! elements (`APPLY_INDEX_MAP`, `GATHER`) — and the three read mechanisms of
//! `../../../README.md` §5.9, with the ROM table and context beneath the
//! third.
//!
//! This module holds what those files share: the evaluation of a map at one
//! entry (§3.3), the address per layout (§3.2), and the owner bundle a read
//! rule takes its `store` port from.
//!
//! **CONST on a composed coordinate.** `../../README.md` §4 step 4 asks CONST
//! "per composed coordinate". Phase 1 builds that coordinate as a term —
//! the batch template substituted into μ, through the arena's constructors
//! (§1.1) — and asks CONST on it. How often depends on the map. An affine μ
//! sends entry j to `μ(t) + Σ_a c_a·s_a·j_a`, a literal offset away from
//! entry 0, so one question per owner axis settles the whole axis; a μ with
//! a `div` or `mod` step — the reshape step of §3.3 — has no such offset and
//! is asked once per entry. That is exactly the rule `../README.md` §3.3
//! states, and it decides the case the old leaf folding missed: a
//! composition that is constant although a leaf of it is not.

pub(crate) mod apply_index_map;
pub(crate) mod enum_coords;
pub(crate) mod gather;
pub(crate) mod read_direct;
pub(crate) mod read_key_mux;
pub(crate) mod read_rom;
pub(crate) mod rom_bind;
pub(crate) mod rom_close_exact;

use crate::phase0::desc::ValueKey;
use crate::phase0::term::{TermArena, TermId};

use crate::phase1::build::Builder;
use crate::phase1::graph::{ReqId, Target};
use crate::phase1::interface::{
    Batch, Capacity, Interface, LayoutKind, Live, Mu, MuForm, OwnerRef, RefIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::{Condition, Question};
use crate::phase1::rules::Candidate;
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

/// The value of one coordinate or address at one entry: a compile-time
/// constant, or a wire (`README.md` §3.1, §3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Composed {
    Literal(Capacity),
    Wire,
}

impl Composed {
    pub fn literal(self) -> Option<Capacity> {
        match self {
            Composed::Literal(value) => Some(value),
            Composed::Wire => None,
        }
    }
}

/// The batch coordinates of one entry (`README.md` §3.1: "a constant where
/// the coordinate term is answered CONST, a wire otherwise").
///
/// CONST is asked on the leaf every entry of an axis shares — the template's
/// base for a range batch, the entry's own coordinate for a general one — so
/// a range batch costs one question per axis (§3.3). A decided leaf is put
/// back as a literal, so the entry's coordinate `t + s·j` folds in the arena
/// and needs no question of its own.
pub fn entry_coordinates(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    batch: &Batch,
    at: Capacity,
    epoch: u32,
    under: Condition,
) -> Vec<Composed> {
    let terms = entry_coordinate_terms(builder, candidate, batch, at, epoch, under);
    terms
        .into_iter()
        .map(|term| match builder.terms.constant_value(term) {
            Some(value) => Composed::Literal(value),
            None => Composed::Wire,
        })
        .collect()
}

/// [`entry_coordinates`] with the terms kept: what the composition of
/// [`Composition`] substitutes into μ.
pub fn entry_coordinate_terms(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    batch: &Batch,
    at: Capacity,
    epoch: u32,
    under: Condition,
) -> Vec<TermId> {
    let bases = batch.axis_bases(&mut builder.terms, at);
    let mut decided: Vec<Option<Capacity>> = Vec::with_capacity(bases.len());
    for base in bases {
        builder.oracle.stats.batch_coordinates += 1;
        let value = match builder.terms.constant_value(base) {
            Some(value) => Some(value),
            None => match builder.ask_for(candidate, Question::Const(base), epoch, under) {
                Verdict::Constant(value) => Some(value),
                _ => None,
            },
        };
        if value.is_some() {
            builder.oracle.stats.constant_coordinates += 1;
        }
        decided.push(value);
    }
    let mut out = batch.coordinate_terms(&mut builder.terms, at);
    // A decided leaf is substituted, so the arena folds the entry's
    // coordinate and every expression built over it. For a range batch the
    // leaf is the axis's base, and the entry's coordinate is that base plus
    // the literal `s_a·j_a`.
    let Batch::Range { axes, .. } = batch else {
        return out;
    };
    let slot = batch.slot(at);
    for (axis, value) in decided.into_iter().enumerate() {
        let (Some(value), Some(template)) = (value, axes.get(axis)) else {
            continue;
        };
        let index = slot.get(axis).copied().unwrap_or(0);
        out[axis] = builder.terms.constant(value + template.stride * index);
    }
    out
}

/// How a read resolves the composed coordinates of a batch: the batch
/// template substituted into the map μ of a `Ref`, or the coordinate itself
/// for the identity map (`../README.md` §4 step 4, `apply_index_map.md`).
///
/// The number of CONST questions follows §3.3: one per owner axis for an
/// affine μ, one per entry through a reshape step. The affine case is the
/// one that needs an argument.
///
/// A coordinate whose leaf CONST decided is put back as a literal before the
/// substitution, so every composed term is built over leaves that are either
/// literals or genuinely not constant. An affine μ is a sum of `k·i_a` and
/// literals, and the arena folds `k·literal` and a sum of literals, so the
/// composed term is a literal exactly when every leaf it mentions with a
/// non-zero coefficient is one — and a zero coefficient folds the leaf away
/// altogether. Hence, for an affine μ, a composed term that is not already a
/// literal mentions a leaf that is not constant, and so is not constant
/// either: the verdict is the same at every entry of the axis. The question
/// is asked at entry 0 and the later entries read their own term, which the
/// arena folded or did not fold for the same reason.
///
/// A `div` or a `mod` breaks that argument — `x div 4` can be constant over
/// an interval where `x` is not — so a μ with a reshape step is asked once
/// per entry, which is what §3.3 says.
pub struct Composition {
    /// Per entry, per owner axis: the composed coordinate.
    entries: Vec<Vec<Composed>>,
}

impl Composition {
    /// Resolve the composition of `batch` with `reference` over the whole
    /// batch.
    pub fn resolve(
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        reference: Option<&RefIface>,
        batch: &Batch,
        epoch: u32,
        under: Condition,
    ) -> Option<Composition> {
        let affine = reference
            .map(|reference| reference.map.iter().all(is_affine))
            .unwrap_or(true);
        let mut entries = Vec::with_capacity(batch.len().max(0) as usize);
        for at in 0..batch.len() {
            let coordinates =
                entry_coordinate_terms(builder, candidate, batch, at, epoch, under);
            let composed = match reference {
                None => coordinates,
                Some(reference) => reference
                    .map
                    .iter()
                    .map(|expression| {
                        substitute(&mut builder.terms, expression, &coordinates, &reference.params)
                    })
                    .collect::<Option<Vec<TermId>>>()?,
            };
            // An affine map is asked once per owner axis, at entry 0; every
            // later entry reads the verdict off the term, which the arena
            // folded to a literal when the leaves were decided.
            let ask = !affine || at == 0;
            let mut values = Vec::with_capacity(composed.len());
            for term in composed {
                values.push(match builder.terms.constant_value(term) {
                    Some(value) => Composed::Literal(value),
                    None if ask => {
                        match builder.ask_for(candidate, Question::Const(term), epoch, under) {
                            Verdict::Constant(value) => Composed::Literal(value),
                            _ => Composed::Wire,
                        }
                    }
                    None => Composed::Wire,
                });
            }
            entries.push(values);
        }
        Some(Composition { entries })
    }

    /// The owner coordinates of entry `at`.
    pub fn at(&self, at: Capacity) -> &[Composed] {
        self.entries
            .get(at.max(0) as usize)
            .map(|entry| entry.as_slice())
            .unwrap_or(&[])
    }
}

/// Whether a μ expression is affine in the batch coordinates, so that the
/// composition of a range batch with it is one question per owner axis.
fn is_affine(expression: &Mu) -> bool {
    match expression {
        Mu::Const(_) | Mu::Coord(_) | Mu::Param(_) => true,
        Mu::Add(left, right) | Mu::Sub(left, right) => is_affine(left) && is_affine(right),
        Mu::Scale(_, inner) => is_affine(inner),
        Mu::ScaleParam(..) | Mu::DivConst(..) | Mu::ModConst(..) => false,
        Mu::DivParam(..) | Mu::ModParam(..) => false,
    }
}

/// One μ expression with the entry's coordinate terms substituted, built
/// through the arena (`README.md` §3.3, "Evaluation at entry j").
///
/// `None` for a construct outside Φ's term language: a product of two
/// non-constants (`π_m × e`) and a division by a wire (`e div π_m`). A read
/// through such a map has run-time addresses whatever the entry, so the
/// caller treats every entry as a wire.
pub fn substitute(
    terms: &mut TermArena,
    expression: &Mu,
    coordinates: &[TermId],
    params: &[TermId],
) -> Option<TermId> {
    Some(match expression {
        Mu::Const(value) => terms.constant(*value),
        Mu::Coord(axis) => *coordinates.get(*axis)?,
        Mu::Param(at) => *params.get(*at)?,
        Mu::Add(left, right) => {
            let a = substitute(terms, left, coordinates, params)?;
            let b = substitute(terms, right, coordinates, params)?;
            terms.add(a, b)
        }
        Mu::Sub(left, right) => {
            let a = substitute(terms, left, coordinates, params)?;
            let b = substitute(terms, right, coordinates, params)?;
            terms.sub(a, b)
        }
        Mu::Scale(factor, inner) => {
            let a = substitute(terms, inner, coordinates, params)?;
            terms.mul_const(*factor, a)
        }
        Mu::DivConst(inner, divisor) => {
            let a = substitute(terms, inner, coordinates, params)?;
            terms.div_const(a, *divisor)?
        }
        Mu::ModConst(inner, modulus) => {
            let a = substitute(terms, inner, coordinates, params)?;
            terms.mod_const(a, *modulus)?
        }
        // §3.1 has no product of two non-constant terms and no division by a
        // term, so these three have no term and no CONST answer.
        Mu::ScaleParam(..) | Mu::DivParam(..) | Mu::ModParam(..) => return None,
    })
}

/// The address of one entry in its owner (`README.md` §3.2): the slot index
/// of the capacity box for a box layout, the logical key for a compact one,
/// and the rank-1 coordinate for a ranked or keyed one.
///
/// `strides` are the key strides `S^o` the `Ref` carries, or the layout's own
/// for the identity map.
pub fn address(owner: &OwnerRef, strides: &[Capacity], coordinates: &[Composed]) -> Composed {
    match &owner.layout {
        LayoutKind::Box(grid) => {
            let mut capacity_strides = vec![1 as Capacity; grid.len()];
            for axis in (0..grid.len().saturating_sub(1)).rev() {
                capacity_strides[axis] = capacity_strides[axis + 1] * grid[axis + 1];
            }
            linear(&capacity_strides, coordinates)
        }
        LayoutKind::Compact(_) => linear(strides, coordinates),
        // Rank 1: the address is the logical index itself.
        LayoutKind::Ranked { .. } | LayoutKind::Keyed { .. } => {
            coordinates.first().copied().unwrap_or(Composed::Wire)
        }
    }
}

/// `Σ_a S_a · κ_a`, a literal when every coordinate is one.
fn linear(strides: &[Capacity], coordinates: &[Composed]) -> Composed {
    let mut total: Capacity = 0;
    for (axis, coordinate) in coordinates.iter().enumerate() {
        let Some(value) = coordinate.literal() else {
            return Composed::Wire;
        };
        total += strides.get(axis).copied().unwrap_or(1) * value;
    }
    Composed::Literal(total)
}

/// The key strides of an owner for the identity map: the capacity strides of
/// a box, and `∏_{b>a} dims_u[b]` of a compact bundle, literal by CONST.
pub fn identity_strides(
    builder: &Builder<'_>,
    owner: &OwnerRef,
    dims: &[TermId],
) -> Option<Vec<Capacity>> {
    match &owner.layout {
        LayoutKind::Box(grid) => {
            let mut strides = vec![1 as Capacity; grid.len()];
            for axis in (0..grid.len().saturating_sub(1)).rev() {
                strides[axis] = strides[axis + 1] * grid[axis + 1];
            }
            Some(strides)
        }
        LayoutKind::Compact(_) => {
            let mut strides = vec![1 as Capacity; dims.len()];
            for axis in (0..dims.len().saturating_sub(1)).rev() {
                let extent = builder.terms.constant_value(dims[axis + 1])?;
                strides[axis] = strides[axis + 1] * extent;
            }
            Some(strides)
        }
        LayoutKind::Ranked { .. } | LayoutKind::Keyed { .. } => {
            (dims.len() == 1).then(|| vec![1])
        }
    }
}

/// The strides a read request evaluates its address with: the `Ref`'s own for
/// a coordinate form, the owner's for the identity map and the key form.
pub fn strides_of(
    builder: &Builder<'_>,
    reference: Option<&RefIface>,
    owner: &OwnerRef,
    owner_dims: &[TermId],
) -> Option<Vec<Capacity>> {
    match reference.map(|reference| &reference.form) {
        Some(MuForm::Coordinate { strides }) => Some(strides.clone()),
        Some(MuForm::Key) => Some(vec![1]),
        None => identity_strides(builder, owner, owner_dims),
    }
}

/// The concrete Store-family requirement an owner node answers, which is the
/// `store` port of every read rule (`read_direct.md`, `read_key_mux.md`,
/// `read_rom.md`, `rom_bind.md`).
///
/// The owner is named by (value, family, capacity, layout), so the row of
/// that value for that family holds exactly one entry matching it.
pub fn store_port(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    request: &RowRequest,
    owner: &OwnerRef,
) -> Result<Option<(ReqId, u32)>, Reject> {
    let psi = builder.psi(&owner.value);
    let condition = builder.child_under_g(candidate, request, psi, &owner.value.encode())?;
    let row = builder.child(
        request,
        Target::Value(owner.value.clone()),
        owner.family,
        condition,
    )?;
    for entry in &row.entries {
        if matches(&entry.interface, owner) {
            return Ok(Some((entry.requirement, entry.epoch)));
        }
    }
    Ok(None)
}

/// Whether a concrete interface is the owner node named.
pub fn matches(interface: &Interface, owner: &OwnerRef) -> bool {
    let Some(candidate) = owner_of(interface, &owner.value) else {
        return false;
    };
    &candidate == owner
}

/// The owner a Store-family interface names.
pub fn owner_of(interface: &Interface, value: &ValueKey) -> Option<OwnerRef> {
    let (capacity, layout, canon) = match interface {
        Interface::Store(store) => (store.capacity, store.lambda.layout.clone(), false),
        Interface::Masked { capacity, lambda, .. } => (*capacity, lambda.layout.clone(), false),
        Interface::Packed { capacity, lambda, canon, .. } => {
            (*capacity, lambda.layout.clone(), *canon)
        }
        _ => return None,
    };
    Some(OwnerRef {
        value: value.clone(),
        family: interface.family(),
        capacity,
        layout,
        canon,
    })
}

/// The `Values` interface a read mechanism answers a `Fetch` with.
pub fn values(builder: &Builder<'_>, value: &ValueKey, batch: &Batch) -> Option<Interface> {
    let dtype = builder.array(value).map(|array| array.dtype).or_else(|| {
        builder.scalar(value).map(|scalar| scalar.dtype)
    })?;
    Some(Interface::Values { dtype, queries: batch.len() })
}

/// Whether every entry of the batch is live by construction, which is what
/// lets a read wire its results through without a flag (`enum_coords.md`).
pub fn always_live(batch: &Batch) -> bool {
    matches!(batch.live(), Live::InRange) && batch.len() > 0
}

/// `w_key = bits(B_tot)`, the width of every address of a context
/// (`README.md` §3.2).
pub fn key_width(owner: &OwnerRef) -> u32 {
    crate::phase1::build::bits_of_capacity(owner.slots())
}
