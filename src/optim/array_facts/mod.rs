//! Array facts: what every element of an array, or of a part of it,
//! satisfies, kept once per array instead of once per element (design and
//! soundness arguments: `docs/array-facts.md`).
//!
//! A fact describes **one array value**, never a variable or a storage, so
//! no write can make it false. The frontend has two kinds of array value
//! (`builder/store.rs`):
//!
//! * a **cell list**: the cells of an array whose cells the frontend knows,
//!   as SSA values, in row-major order. A source states a [`Segmentation`]
//!   over a list; the store records, for every runtime cell, the lists it is
//!   in and its position there. A write `a[0] = x` gives `a` a different list
//!   (its cell 0 is `x`), about which the old facts say nothing. Views,
//!   slices, copies and reshapes share cells, so element facts follow the
//!   cells with no code in those operators; order and permutation facts
//!   follow a view that is a run of a stated list, in order or reversed
//!   ([`ArrayFacts::list_facts`]);
//! * an **array version**: the IR statement (`ArrayInit` / `ArrayWrite`) an
//!   array's cells are after a write at a runtime position
//!   ([`ArrayFacts::state_version`]). A write makes a new version, whose facts
//!   the builder derives from the old version's by the write rule.
//!
//! **Where a fact holds.** Every stated fact carries a [`FactGuard`], as the
//! scalar facts of `FactStack` do: `Always` for a fact that holds on every
//! path (a definition, a check at entry), or `When(atoms)` for one derived
//! from what is known where the code runs (its execution predicate's
//! conjuncts). A query names the context it asks for; it sees a fact only
//! when the fact's atoms are among the context's, so a fact never justifies
//! anything on a path where it is not known to hold.
//!
//! Every fact recorded here must satisfy the trust rule: a constraint
//! enforces it on every accepted witness (where its guard holds), or it holds
//! by the definition of the cells. Each source documents its argument where
//! it states a fact. Reasoning is lookup and interval arithmetic only: no
//! solver, no quantifier. "For every cell" is the segmentation itself.

pub mod domain;
pub mod segmentation;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

pub use domain::{transfer, Affine, Bound, ElemFacts, NoScalars, Order, Range, Scalars};
pub use segmentation::{Segment, Segmentation};

pub use crate::optim::predicates::FactGuard;
use crate::types::{StmtId, Value};

/// One cell of a list: a compile-time integer, a compile-time float (by its
/// bits: for identity and order only, it has no integer range), or the
/// statement computing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cell {
    Const(i64),
    FloatConst(u64),
    Stmt(StmtId),
}

impl Cell {
    /// The cell a scalar value is; `None` for anything else (or a scalar
    /// with neither a constant nor a statement).
    pub fn of(v: &Value) -> Option<Cell> {
        match v {
            Value::Integer(_) | Value::Boolean(_) => match (v.int_val(), v.stmt_id()) {
                (Some(c), _) => Some(Cell::Const(c)),
                (None, Some(s)) => Some(Cell::Stmt(s)),
                (None, None) => None,
            },
            Value::Float(_) => match (v.float_val(), v.stmt_id()) {
                (Some(f), _) => Some(Cell::FloatConst(f.to_bits())),
                (None, Some(s)) => Some(Cell::Stmt(s)),
                (None, None) => None,
            },
            _ => None,
        }
    }

    /// Cells of a list of values; `None` if any value is not a scalar cell.
    pub fn list(values: &[Value]) -> Option<Vec<Cell>> {
        values.iter().map(Cell::of).collect()
    }
}

/// Whether the ablation `ZINNIA_ABLATE_ARRAY_FACTS=1` turns array facts off
/// (`docs/ablations.md`): no query sees any (the compiler then decides as if
/// nothing were known about whole arrays).
pub fn ablated() -> bool {
    std::env::var("ZINNIA_ABLATE_ARRAY_FACTS").is_ok_and(|v| v == "1")
}

/// Whether a fact stated under `guard` may be used in `context`: it holds on
/// every path, or every conjunct it was learned under is a conjunct of the
/// context. Nothing guarded is used where the code never runs.
pub fn holds_in(guard: &FactGuard, context: &FactGuard) -> bool {
    match (guard, context) {
        (FactGuard::Always, _) => true,
        (FactGuard::When(atoms), FactGuard::When(here)) => {
            let mut rest = here.iter();
            atoms.iter().all(|a| rest.by_ref().any(|b| b == a))
        }
        _ => false,
    }
}

/// Identifies a stated list (index into [`ArrayFacts::lists`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ListId(u32);

#[derive(Clone, Debug)]
struct StatedList {
    cells: Vec<Cell>,
    facts: Segmentation,
    guard: FactGuard,
}

/// The facts stated about cell lists and array versions, with an index from
/// each runtime cell to the lists it is in.
#[derive(Clone, Debug, Default)]
pub struct ArrayFacts {
    lists: Vec<StatedList>,
    /// For each runtime cell: the lists it belongs to and its position there,
    /// in the order they were stated (so every walk is deterministic).
    members: HashMap<StmtId, Vec<(ListId, u32)>>,
    /// The facts of array versions (`ArrayInit` / `ArrayWrite` statements),
    /// in the order they were stated.
    versions: HashMap<StmtId, Vec<(Segmentation, FactGuard)>>,
}

impl ArrayFacts {
    pub fn new() -> ArrayFacts {
        ArrayFacts::default()
    }

    /// State `facts` about `cells` (the cells of one array, row-major), where
    /// `guard` holds. Nothing is recorded when the facts state nothing, when
    /// the list has no runtime cell (constants carry their own facts:
    /// [`constant_facts`]) or when the guard is `Never`.
    pub fn state(&mut self, cells: Vec<Cell>, facts: Segmentation, guard: FactGuard) -> Option<ListId> {
        assert_eq!(cells.len(), facts.len(), "array facts: a segmentation must cover exactly the list's cells");
        if facts.is_top() || guard == FactGuard::Never || !cells.iter().any(|c| matches!(c, Cell::Stmt(_))) {
            return None;
        }
        let id = ListId(self.lists.len() as u32);
        for (p, cell) in cells.iter().enumerate() {
            if let Cell::Stmt(s) = cell {
                self.members.entry(*s).or_default().push((id, p as u32));
            }
        }
        self.lists.push(StatedList { cells, facts, guard });
        Some(id)
    }

    /// State `facts` about the array version `version` (an `ArrayInit` or
    /// `ArrayWrite` statement), where `guard` holds.
    pub fn state_version(&mut self, version: StmtId, facts: Segmentation, guard: FactGuard) {
        if facts.is_top() || guard == FactGuard::Never {
            return;
        }
        self.versions.entry(version).or_default().push((facts, guard));
    }

    /// The facts of the array version `version` that hold in `context`.
    pub fn version_facts(&self, version: StmtId, context: &FactGuard) -> Vec<&Segmentation> {
        self.versions
            .get(&version)
            .into_iter()
            .flatten()
            .filter(|(_, guard)| holds_in(guard, context))
            .map(|(facts, _)| facts)
            .collect()
    }

    /// Every stated fact about one cell that holds in `context`, one per list
    /// it is in (each holds; a consumer may use any of them).
    pub fn cell_facts(&self, cell: Cell, s: &dyn Scalars, context: &FactGuard) -> Vec<ElemFacts> {
        match cell {
            Cell::Const(c) => vec![ElemFacts::range(Range::point(c))],
            Cell::FloatConst(_) => Vec::new(),
            Cell::Stmt(stmt) => self
                .members
                .get(&stmt)
                .into_iter()
                .flatten()
                .filter_map(|&(list, p)| {
                    let stated = &self.lists[list.0 as usize];
                    holds_in(&stated.guard, context).then(|| stated.facts.at(p as usize, s))
                })
                .collect(),
        }
    }

    /// The range of one cell: the meet of the ranges its facts give.
    pub fn cell_range(&self, cell: Cell, s: &dyn Scalars, context: &FactGuard) -> Range {
        self.cell_facts(cell, s, context).iter().fold(Range::TOP, |acc, f| acc.meet(&f.range, s))
    }

    /// The facts that hold in `context` about `cells` as an array of their
    /// own: the facts of every stated list of which `cells` is a run of
    /// consecutive positions, in order or reversed (a slice, a reversed view,
    /// the list itself), in the order the lists were stated, and the facts
    /// the cells carry as constants. Lists are found through the memberships
    /// of the run's first runtime cell.
    pub fn list_facts(&self, cells: &[Cell], s: &dyn Scalars, context: &FactGuard) -> Vec<Segmentation> {
        if let Some(constant) = constant_facts(cells) {
            return vec![constant];
        }
        let Some((k, anchor)) = cells.iter().enumerate().find_map(|(k, c)| match c {
            Cell::Stmt(stmt) => Some((k, *stmt)),
            _ => None,
        }) else {
            return Vec::new();
        };
        let n = cells.len();
        let mut out = Vec::new();
        for &(list, p) in self.members.get(&anchor).into_iter().flatten() {
            let stated = &self.lists[list.0 as usize];
            if !holds_in(&stated.guard, context) {
                continue;
            }
            let p = p as usize;
            // In order: cells[i] = stated[p - k + i].
            if p >= k && p - k + n <= stated.cells.len() && stated.cells[p - k..p - k + n] == *cells {
                out.push(stated.facts.slice(p - k, p - k + n, s));
                continue;
            }
            // Reversed: cells[i] = stated[p + k - i].
            if p + k < stated.cells.len() && p + k + 1 >= n {
                let start = p + k + 1 - n;
                let run = &stated.cells[start..=p + k];
                if run.iter().rev().eq(cells.iter()) {
                    out.push(stated.facts.slice(start, p + k + 1, s).reversed(s));
                }
            }
        }
        out
    }
}

/// The facts a list of compile-time constants has by inspection (the tier of
/// constants, applied to a whole array): integer cells give their range, an
/// arithmetic progression its affine formula, and both kinds their order.
/// `None` when a cell is not a constant.
pub fn constant_facts(cells: &[Cell]) -> Option<Segmentation> {
    if cells.is_empty() {
        return None;
    }
    let ints: Option<Vec<i64>> = cells.iter().map(|c| if let Cell::Const(v) = c { Some(*v) } else { None }).collect();
    if let Some(ints) = ints {
        let (lo, hi) = (*ints.iter().min()?, *ints.iter().max()?);
        let pairs = || ints.windows(2).map(|w| (w[0], w[1]));
        let order = order_of(pairs());
        let step = if ints.len() > 1 { ints[1].checked_sub(ints[0]) } else { Some(0) };
        let affine = step
            .filter(|&d| ints.windows(2).all(|w| w[1].checked_sub(w[0]) == Some(d)))
            .map(|step| Affine { base: ints[0], step });
        let facts = ElemFacts { range: Range::new(lo, hi), order, affine };
        return Some(Segmentation::uniform(ints.len(), facts));
    }
    let floats: Option<Vec<f64>> =
        cells.iter().map(|c| if let Cell::FloatConst(b) = c { Some(f64::from_bits(*b)) } else { None }).collect();
    let floats = floats?;
    if floats.iter().any(|f| f.is_nan()) {
        return None;
    }
    // Floats are fixed-point field elements: rounding to the grid keeps
    // `a <= b` but may make two different constants equal, so a strict order
    // is kept only as a non-strict one.
    let order = order_of(floats.windows(2).map(|w| (w[0], w[1]))).map(Order::non_strict);
    Some(Segmentation::uniform(floats.len(), ElemFacts { order, ..ElemFacts::TOP }))
}

/// The strongest order every consecutive pair satisfies (a list of one cell
/// is in every order; `Increasing` is reported).
fn order_of<T: PartialOrd>(pairs: impl Iterator<Item = (T, T)> + Clone) -> Option<Order> {
    let all = |f: fn(&T, &T) -> bool| pairs.clone().all(|(a, b)| f(&a, &b));
    if all(|a, b| a < b) {
        Some(Order::Increasing)
    } else if all(|a, b| a > b) {
        Some(Order::Decreasing)
    } else if all(|a, b| a <= b) {
        Some(Order::NonDecreasing)
    } else if all(|a, b| a >= b) {
        Some(Order::NonIncreasing)
    } else {
        None
    }
}
