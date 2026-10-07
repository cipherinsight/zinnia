//! Facts about the whole IR graph that the per-statement lowering uses
//! before the statements that establish them are reached.
//!
//! Every fact here is read off the IR's statements, never off witness
//! values, so key generation and proving take the same decisions.
//!
//! * **Planned bounds.** `AssertU128Lt(x, hi)` with a constant `hi` is a
//!   constraint of the circuit wherever it appears in the statement list: a
//!   circuit has no control flow, and every IR statement is emitted. When an
//!   earlier statement needs `x` inside the 64-bit window (an input-domain
//!   check `x >= i64::MIN`, a comparison), the bound check can be emitted
//!   there instead: it proves `0 <= x < hi`, which lies inside the window,
//!   and the later `AssertU128Lt` then finds it proven and emits nothing.
//!   The set of accepted witnesses is unchanged.
//! * **Deferred products** (see `Halo2BaseSynthesizer::mul_i`, `add_i`).
//! * **Asserted comparisons.** A comparison whose only use is an `Assert`
//!   needs no result bit: an order is range-checked directly, `a != b` is an
//!   inverse, `a == b` a copy constraint (see
//!   `Halo2BaseSynthesizer::assert_relation`, `assert_distinct`,
//!   `assert_equal`).
//! * **Bit hints.** A prover hint that an `AssertBit` checks is assigned
//!   inside its bit gate (see `Halo2BaseSynthesizer::prover_hint`).

use std::collections::{HashMap, HashSet};

use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::ValueId;

#[derive(Default)]
pub(super) struct Plan {
    /// `x -> hi` for every `AssertU128Lt(x, hi)` with a constant `hi > 0`
    /// (the smallest when there are several).
    pub bounds: HashMap<ValueId, i128>,
    /// Statement ids of the `MulI` products whose only use is one `AddI`
    /// or the subtrahend of one `SubI`: the sum is emitted as one
    /// multiply-add gate there, and the product is never a cell.
    pub deferred_products: HashSet<u32>,
    /// Statement ids of the comparisons (`<`, `<=`, `>`, `>=`, `!=`, `==`,
    /// key orders) whose only use is `Assert`.
    pub asserted_comparisons: HashSet<u32>,
    /// Statement ids of the prover hints some `AssertBit` checks directly:
    /// each is loaded inside its own bit gate.
    pub bit_hints: HashSet<u32>,
    /// Select chains over the equality bits of one bounded index (see
    /// [`OneHotChain`]), by the statement id of the chain's last select.
    pub onehot_chains: HashMap<u32, OneHotChain>,
    /// The other selects of those chains: they get no cell.
    pub onehot_links: HashSet<u32>,
    /// Statement ids of the `BitAndI` with a constant single-bit mask `2^j`
    /// whose every use is a comparison with a constant: such a comparison
    /// is the bit (two-valued facts), so `x & 2^j = 2^j·bit` needs no cell.
    pub cellless_bit_masks: HashSet<u32>,
    /// Asserted `x >= 0` with a later `sqrt(x)` (`crate::prove::guards`;
    /// see `Halo2BaseSynthesizer::assert_nonneg_by_root`).
    pub root_guards: HashSet<u32>,
    /// Asserted `d != 0` with a later division by `d` (`crate::prove::guards`;
    /// see `Halo2BaseSynthesizer::assert_distinct`).
    pub divisor_guards: HashSet<u32>,
    /// Private `ReadInteger` statements with an `AssertU128Lt(x, hi)` of a
    /// constant `hi > 0` (the smallest): the input is loaded inside the gate
    /// of that check (see `Halo2BaseSynthesizer::read_bounded_input`).
    pub bounded_inputs: HashMap<u32, i128>,
}

/// `select([r == k_m], x_m, ... select([r == k_1], x_1, x_0))` where
/// `AssertU128Lt(r, n)` holds somewhere in the circuit and `{k_1..k_m}` is
/// `{0..n-1}` without one value `k_0`: exactly one of `[r == k_0], ...,
/// [r == k_m]` is 1, so the chain is the inner product
/// `sum_i [r == k_i] x_i` (one cell triple per term instead of a select gate
/// pair per link).
#[derive(Clone, Debug)]
pub(super) struct OneHotChain {
    pub root: ValueId,
    pub missing: i64,
    pub default: ValueId,
    /// `(condition, value)` per link, innermost first.
    pub terms: Vec<(ValueId, ValueId)>,
}

impl Plan {
    fn with_onehot_chains(mut self, ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> Plan {
        let (chains, links) = onehot_chains(ir, uses, &self.bounds);
        self.onehot_chains = chains;
        self.onehot_links = links;
        self
    }

    pub fn of(ir: &IRGraph) -> Plan {
        let uses = uses_of(ir);
        Plan {
            bounds: planned_bounds(ir),
            deferred_products: deferred_products(ir, &uses),
            asserted_comparisons: asserted_comparisons(ir, &uses),
            bit_hints: bit_hints(ir, &uses),
            onehot_chains: HashMap::new(),
            onehot_links: HashSet::new(),
            cellless_bit_masks: cellless_bit_masks(ir, &uses),
            root_guards: crate::prove::guards::root_guards(ir),
            divisor_guards: crate::prove::guards::divisor_guards(ir),
            bounded_inputs: bounded_inputs(ir),
        }
        .with_onehot_chains(ir, &uses)
    }
}

/// The statements that use each statement's value, by statement id.
fn uses_of(ir: &IRGraph) -> HashMap<u32, Vec<u32>> {
    let mut uses: HashMap<u32, Vec<u32>> = HashMap::new();
    for stmt in &ir.stmts {
        for arg in &stmt.arguments {
            uses.entry(*arg).or_default().push(stmt.stmt_id);
        }
    }
    uses
}

/// The one statement that uses `stmt`'s value, if exactly one does.
fn only_user<'a>(ir: &'a IRGraph, uses: &HashMap<u32, Vec<u32>>, stmt: u32) -> Option<&'a crate::ir::IRStatement> {
    match uses.get(&stmt).map(Vec::as_slice) {
        Some([user]) => ir.stmts.get(*user as usize),
        _ => None,
    }
}

fn asserted_comparisons(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    ir.stmts
        .iter()
        .filter(|stmt| {
            matches!(
                stmt.ir,
                IR::LtI
                    | IR::LteI
                    | IR::GtI
                    | IR::GteI
                    | IR::LtF
                    | IR::LteF
                    | IR::GtF
                    | IR::GteF
                    | IR::EqI
                    | IR::EqF
                    | IR::NeI
                    | IR::NeF
                    | IR::EqI
                    | IR::EqF
                    | IR::KeyLess { .. }
                    | IR::PackedKeyLess { .. }
            )
        })
        .filter(|stmt| only_user(ir, uses, stmt.stmt_id).is_some_and(|user| matches!(user.ir, IR::Assert)))
        .map(|stmt| stmt.stmt_id)
        .collect()
}

fn bit_hints(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    ir.stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::Hint { .. }))
        .filter(|stmt| {
            uses.get(&stmt.stmt_id).is_some_and(|users| {
                users.iter().any(|user| ir.stmts.get(*user as usize).is_some_and(|u| matches!(u.ir, IR::AssertBit)))
            })
        })
        .map(|stmt| stmt.stmt_id)
        .collect()
}

fn constant_int(ir: &IRGraph, stmt: u32) -> Option<i64> {
    match ir.stmts.get(stmt as usize).map(|s| &s.ir) {
        Some(IR::ConstantInt { value }) => Some(*value),
        _ => None,
    }
}

fn planned_bounds(ir: &IRGraph) -> HashMap<ValueId, i128> {
    let mut bounds: HashMap<ValueId, i128> = HashMap::new();
    for stmt in &ir.stmts {
        if !matches!(stmt.ir, IR::AssertU128Lt) || stmt.arguments.len() != 2 {
            continue;
        }
        let Some(hi) = constant_int(ir, stmt.arguments[1]).filter(|hi| *hi > 0) else { continue };
        let Some(x) = ir.stmts.get(stmt.arguments[0] as usize) else { continue };
        let entry = bounds.entry(x.value_id).or_insert(hi as i128);
        *entry = (*entry).min(hi as i128);
    }
    bounds
}

/// `x -> hi` for private integer inputs `x` with an `AssertU128Lt(x, hi)`,
/// `hi` a positive constant (the smallest when there are several).
fn bounded_inputs(ir: &IRGraph) -> HashMap<u32, i128> {
    let mut bounds: HashMap<u32, i128> = HashMap::new();
    for stmt in &ir.stmts {
        let (IR::AssertU128Lt, [x, hi]) = (&stmt.ir, stmt.arguments.as_slice()) else { continue };
        let Some(hi) = constant_int(ir, *hi).filter(|hi| *hi > 0) else { continue };
        if matches!(ir.stmts.get(*x as usize).map(|s| &s.ir), Some(IR::ReadInteger { is_public: false, .. })) {
            let entry = bounds.entry(*x).or_insert(hi as i128);
            *entry = (*entry).min(hi as i128);
        }
    }
    bounds
}

/// A `MulI` qualifies when every statement that reads its value is an
/// `AddI`/`AddF` (either operand, not both) or a `SubI`/`SubF` (the
/// subtrahend). Each reader then emits the product inside its own multiply-add gate:
/// the product's own gate is saved, and every reader's gate has as many
/// cells as the plain addition it replaces. A product with several readers
/// qualifies only when no reader adds it to another qualifying product
/// (such a sum makes one of them a cell, which would be built again for
/// every reader).
fn deferred_products(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    let fused_by = |product: u32, user: u32| -> bool {
        let Some(user) = ir.stmts.get(user as usize) else { return false };
        match (&user.ir, user.arguments.as_slice()) {
            // `add_f`/`sub_f` are `add_i`/`sub_i` on the Q encodings.
            (IR::AddI | IR::AddF, [a, b]) => a != b,
            (IR::SubI | IR::SubF, [a, b]) => *b == product && a != b,
            _ => false,
        }
    };
    let candidates: HashSet<u32> = ir
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::MulI))
        .filter(|stmt| {
            uses.get(&stmt.stmt_id)
                .is_some_and(|users| users.iter().all(|&user| fused_by(stmt.stmt_id, user)))
        })
        .map(|stmt| stmt.stmt_id)
        .collect();
    let other_operand = |user: u32, product: u32| -> Option<u32> {
        ir.stmts.get(user as usize)?.arguments.iter().copied().find(|&arg| arg != product)
    };
    candidates
        .iter()
        .copied()
        .filter(|&product| match uses.get(&product).map(Vec::as_slice) {
            Some([_]) => true,
            Some(users) => users
                .iter()
                .all(|&user| other_operand(user, product).is_none_or(|other| !candidates.contains(&other))),
            None => false,
        })
        .collect()
}

/// `(root, k)` when `stmt` is `EqI(root, k)` or `EqI(k, root)` for a constant `k`.
fn equality_with_constant(ir: &IRGraph, stmt: u32) -> Option<(u32, i64)> {
    let st = ir.stmts.get(stmt as usize)?;
    if !matches!(st.ir, IR::EqI) {
        return None;
    }
    match st.arguments.as_slice() {
        [a, b] => match (constant_int(ir, *a), constant_int(ir, *b)) {
            (None, Some(k)) => Some((*a, k)),
            (Some(k), None) => Some((*b, k)),
            _ => None,
        },
        _ => None,
    }
}

/// See [`OneHotChain`]. A chain is found from its last select by following
/// else-branches that are selects used only there; it is kept when it has at
/// least three links (shorter chains cost no more as selects).
fn onehot_chains(
    ir: &IRGraph,
    uses: &HashMap<u32, Vec<u32>>,
    bounds: &HashMap<ValueId, i128>,
) -> (HashMap<u32, OneHotChain>, HashSet<u32>) {
    let is_select = |id: u32| ir.stmts.get(id as usize).is_some_and(|s| matches!(s.ir, IR::SelectI | IR::SelectF));
    let link = |id: u32| -> Option<(u32, i64, u32, u32)> {
        let st = &ir.stmts[id as usize];
        let [c, t, f] = st.arguments.as_slice() else { return None };
        let (root, k) = equality_with_constant(ir, *c)?;
        Some((root, k, *t, *f))
    };
    let mut chains = HashMap::new();
    let mut links = HashSet::new();
    for stmt in &ir.stmts {
        if !is_select(stmt.stmt_id) {
            continue;
        }
        // Not the end when the only user continues the chain on the same root.
        if let Some(user) = only_user(ir, uses, stmt.stmt_id) {
            if is_select(user.stmt_id) && user.arguments.get(2) == Some(&stmt.stmt_id) {
                if let (Some((r1, ..)), Some((r2, ..))) = (link(user.stmt_id), link(stmt.stmt_id)) {
                    if r1 == r2 {
                        continue;
                    }
                }
            }
        }
        let Some((root, _, _, _)) = link(stmt.stmt_id) else { continue };
        let mut members = Vec::new();
        let mut cur = stmt.stmt_id;
        loop {
            match link(cur) {
                Some((r, k, t, f)) if r == root && is_select(cur) => {
                    members.push((cur, k, ir.stmts[t as usize].value_id, ir.stmts[link_cond(ir, cur)].value_id));
                    let continues = is_select(f)
                        && only_user(ir, uses, f).is_some_and(|u| u.stmt_id == cur)
                        && link(f).is_some_and(|(r2, ..)| r2 == root);
                    if !continues {
                        cur = f;
                        break;
                    }
                    cur = f;
                }
                _ => break,
            }
        }
        if members.len() < 3 {
            continue;
        }
        let root_vid = ir.stmts[root as usize].value_id;
        let Some(&hi) = bounds.get(&root_vid) else { continue };
        let ks: HashSet<i64> = members.iter().map(|m| m.1).collect();
        if ks.len() != members.len() || hi as usize != members.len() + 1 || ks.iter().any(|k| *k < 0 || *k as i128 >= hi) {
            continue;
        }
        let Some(missing) = (0..hi as i64).find(|k| !ks.contains(k)) else { continue };
        let terms: Vec<(ValueId, ValueId)> = members.iter().rev().map(|m| (m.3, m.2)).collect();
        for m in &members[1..] {
            links.insert(m.0);
        }
        chains.insert(
            stmt.stmt_id,
            OneHotChain { root: root_vid, missing, default: ir.stmts[cur as usize].value_id, terms },
        );
    }
    (chains, links)
}

fn link_cond(ir: &IRGraph, select: u32) -> usize {
    ir.stmts[select as usize].arguments[0] as usize
}

/// A `BitAndI` of a cell and a constant `2^j` (`j < 63`) whose every user is
/// a comparison (`==`, `!=`, `<`, `<=`, `>`, `>=`) of it with a constant,
/// and which no affine view reads.
fn cellless_bit_masks(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    let roots: HashSet<ValueId> = ir
        .stmts
        .iter()
        .filter_map(|stmt| ir.provenance.affine_of(stmt.value_id))
        .flat_map(|view| view.terms.iter().map(|(root, _)| *root))
        .collect();
    let single_bit = |c: Option<i64>| c.is_some_and(|c| c > 0 && c & (c - 1) == 0);
    ir.stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::BitAndI) && !roots.contains(&stmt.value_id))
        .filter(|stmt| match stmt.arguments.as_slice() {
            [a, b] => single_bit(constant_int(ir, *a)) != single_bit(constant_int(ir, *b))
                && constant_int(ir, *a).is_none() != constant_int(ir, *b).is_none(),
            _ => false,
        })
        .filter(|stmt| {
            let Some(users) = uses.get(&stmt.stmt_id).filter(|u| !u.is_empty()) else { return false };
            users.iter().all(|user| {
                let Some(user) = ir.stmts.get(*user as usize) else { return false };
                let comparison = matches!(user.ir, IR::EqI | IR::NeI | IR::LtI | IR::LteI | IR::GtI | IR::GteI);
                comparison
                    && user.arguments.len() == 2
                    && user.arguments.iter().any(|a| *a != stmt.stmt_id && constant_int(ir, *a).is_some())
            })
        })
        .map(|stmt| stmt.stmt_id)
        .collect()
}
