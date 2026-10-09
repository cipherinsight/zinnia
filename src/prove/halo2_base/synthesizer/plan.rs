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
    pub affine_roots: HashMap<u32, AffineTree>,
    pub affine_links: HashSet<u32>,
    pub dyadic_links: HashMap<u32, u32>,
    pub clips: HashMap<u32, Clip>,
    pub clip_links: HashSet<u32>,
    pub select_adds: HashMap<u32, (ValueId, ValueId)>,
    pub select_add_links: HashSet<u32>,
    pub falsy_casts: HashSet<u32>,
    pub falsy_nots: HashMap<u32, ValueId>,
    pub vector_ops: HashSet<u32>,
    pub vector_links: HashSet<u32>,
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
    pub square_endpoint_guards: HashSet<u32>,
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

#[derive(Clone, Copy, Debug)]
pub(super) struct Clip { pub input: ValueId, pub lo: i64, pub hi: i64 }

#[derive(Clone, Debug)]
pub(super) enum AffineTerm {
    Linear(ValueId, i128),
    Product(ValueId, ValueId),
}

#[derive(Clone, Debug)]
pub(super) struct AffineTree {
    pub terms: Vec<AffineTerm>,
}

impl Plan {
    fn with_onehot_chains(mut self, ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> Plan {
        let (chains, links) = onehot_chains(ir, uses, &self.bounds);
        self.onehot_chains = chains;
        self.onehot_links = links;
        self.select_adds.retain(|id, _| !self.onehot_chains.contains_key(id) && !self.onehot_links.contains(id));
        self.select_add_links = self.select_adds.keys().map(|id| ir.stmts[*id as usize].arguments[1]).collect();
        self.vector_links.retain(|id| uses.get(id).is_some_and(|users|
            users.iter().all(|u| !self.onehot_links.contains(u) && !self.onehot_chains.contains_key(u))));
        self
    }

    pub fn of(ir: &IRGraph) -> Plan {
        let uses = uses_of(ir);
        let (affine_roots, affine_links) = affine_trees(ir, &uses);
        let (clips, clip_links) = constant_clips(ir, &uses);
        let (vector_ops, vector_links) = bitvector_islands(ir, &uses);
        let (mut select_adds, _, falsy_casts, falsy_nots) = conditional_arithmetic(ir, &uses);
        select_adds.retain(|id, _| !affine_roots.contains_key(&ir.stmts[*id as usize].arguments[1]));
        let select_add_links = select_adds.keys().map(|id| ir.stmts[*id as usize].arguments[1]).collect();
        Plan {
            bounds: planned_bounds(ir),
            deferred_products: deferred_products(ir, &uses),
            vector_ops,
            vector_links,
            select_adds,
            select_add_links,
            falsy_casts,
            falsy_nots,
            clips,
            clip_links,
            affine_roots,
            affine_links,
            dyadic_links: dyadic_links(ir, &uses),
            asserted_comparisons: asserted_comparisons(ir, &uses),
            bit_hints: bit_hints(ir, &uses),
            onehot_chains: HashMap::new(),
            onehot_links: HashSet::new(),
            cellless_bit_masks: cellless_bit_masks(ir, &uses),
            root_guards: crate::prove::guards::root_guards(ir),
            divisor_guards: crate::prove::guards::divisor_guards(ir),
            square_endpoint_guards: square_endpoint_guards(ir, &uses),
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

/// Single-consumer exact trees. Never cross a rounded product, a domain check,
/// or a value used by address provenance. Multiplication is folded only when
/// constant-weighted, or a positive product directly inside the inner product.
fn affine_trees(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> (HashMap<u32, AffineTree>, HashSet<u32>) {
    let provenance_roots = observed_affine_roots(ir);
    let arithmetic = |id: u32| matches!(ir.stmts[id as usize].ir,
        IR::AddI | IR::AddF | IR::SubI | IR::SubF | IR::SumF { .. } | IR::MulI);
    let internal = |id: u32| !provenance_roots.contains(&ir.stmts[id as usize].value_id)
        && only_user(ir, uses, id).is_some_and(|s| arithmetic(s.stmt_id));
    let mut roots = HashMap::new();
    let mut links = HashSet::new();
    // Reverse order gives each maximal tree first choice of its internal nodes.
    for root in ir.stmts.iter().rev() {
        if !arithmetic(root.stmt_id) || links.contains(&root.stmt_id) { continue; }
        // A MulI alone is already a single gate (or an existing deferred product).
        if matches!(root.ir, IR::MulI) { continue; }
        let mut stack = vec![(root.stmt_id, 1i128)];
        let mut consumed = Vec::new();
        let mut terms = Vec::new();
        while let Some((id, coefficient)) = stack.pop() {
            let s = &ir.stmts[id as usize];
            let can_expand = id == root.stmt_id || (arithmetic(id) && internal(id));
            if can_expand {
                match (&s.ir, s.arguments.as_slice()) {
                    (IR::AddI | IR::AddF, [a, b]) => {
                        consumed.push(id);
                        stack.push((*b, coefficient)); stack.push((*a, coefficient)); continue;
                    }
                    (IR::SubI | IR::SubF, [a, b]) if coefficient != i128::MIN => {
                        consumed.push(id);
                        stack.push((*b, -coefficient)); stack.push((*a, coefficient)); continue;
                    }
                    (IR::SumF { .. }, args) if !args.is_empty() => {
                        consumed.push(id);
                        stack.extend(args.iter().rev().map(|a| (*a, coefficient))); continue;
                    }
                    (IR::MulI, [a, b]) => {
                        let constant = constant_int(ir, *a).map(|c| (c, *b))
                            .or_else(|| constant_int(ir, *b).map(|c| (c, *a)));
                        if let Some((c, other)) = constant {
                            if let Some(weight) = coefficient.checked_mul(c as i128) {
                                consumed.push(id); stack.push((other, weight)); continue;
                            }
                        } else if coefficient == 1 {
                            consumed.push(id);
                            terms.push(AffineTerm::Product(ir.stmts[*a as usize].value_id, ir.stmts[*b as usize].value_id));
                            continue;
                        }
                    }
                    _ => {}
                }
            }
            terms.push(AffineTerm::Linear(s.value_id, coefficient));
        }
        // A lone binary op gains nothing; retain its existing fact/selection
        // fusion. Trees and SumF benefit from contiguous accumulator layout.
        if consumed.len() < 2 { continue; }
        for id in consumed { if id != root.stmt_id { links.insert(id); } }
        roots.insert(root.stmt_id, AffineTree { terms });
    }
    (roots, links)
}

/// The intermediate floor of a dyadic chain can be deferred only if all its
/// readers apply another positive dyadic divisor. Runtime range proofs must
/// additionally show the skipped signed64 quotient check is implied.
fn dyadic_links(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashMap<u32, u32> {
    let positive_fraction = |id: u32| match ir.stmts[id as usize].ir {
        IR::ConstantFloat { value } => value > 0.0 && value < 1.0 && value.log2().fract() == 0.0,
        _ => false,
    };
    let roots: HashSet<_> = ir.stmts.iter().filter_map(|s| ir.provenance.affine_of(s.value_id))
        .flat_map(|v| v.terms.iter().map(|(root, _)| *root)).collect();
    ir.stmts.iter().filter(|s| matches!(s.ir, IR::MulF) && !roots.contains(&s.value_id))
        .filter(|s| uses.get(&s.stmt_id).is_some_and(|users| !users.is_empty() && users.iter().all(|id| {
            let user = &ir.stmts[*id as usize];
            matches!(user.ir, IR::MulF) && user.arguments.len() == 2
                && user.arguments.iter().any(|a| *a != s.stmt_id && positive_fraction(*a))
        }))).map(|s| {
            let extra = uses[&s.stmt_id].iter().flat_map(|id| ir.stmts[*id as usize].arguments.iter())
                .filter_map(|id| match ir.stmts[*id as usize].ir {
                    IR::ConstantFloat { value } if value > 0.0 && value < 1.0 => Some((-value.log2()) as u32),
                    _ => None,
                }).max().unwrap_or(64);
            (s.stmt_id, extra)
        }).collect()
}

/// At P <= 63, a direct Q input equal to -2^63 cannot pass the signed64
/// quotient bound of its rounded square (2^126 / 2^P >= 2^63). Restrict to
/// ReadFloat so float-cast exact-product shortcuts cannot bypass that bound.
fn square_endpoint_guards(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    let squared: HashSet<_> = ir.stmts.iter().filter_map(|s| match (&s.ir, s.arguments.as_slice()) {
        (IR::MulF, [a, b]) if a == b && matches!(ir.stmts[*a as usize].ir, IR::ReadFloat { .. }) => Some(*a),
        _ => None,
    }).collect();
    ir.stmts.iter().filter(|s| matches!(s.ir, IR::NeI | IR::NeF))
        .filter(|s| only_user(ir, uses, s.stmt_id).is_some_and(|u| matches!(u.ir, IR::Assert)))
        .filter(|s| match s.arguments.as_slice() {
            [a, b] => (squared.contains(a) && constant_int(ir, *b) == Some(i64::MIN))
                || (squared.contains(b) && constant_int(ir, *a) == Some(i64::MIN)),
            _ => false,
        }).map(|s| s.stmt_id).collect()
}

/// Recognize min(max(x, lo), hi), retaining the original signed64 operand
/// domain. The two comparisons and intermediate select must be private to
/// this clamp, including provenance readers.
fn constant_clips(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> (HashMap<u32, Clip>, HashSet<u32>) {
    let roots: HashSet<_> = ir.stmts.iter().filter_map(|s| ir.provenance.affine_of(s.value_id))
        .flat_map(|v| v.terms.iter().map(|(root, _)| *root)).collect();
    let mut clips = HashMap::new(); let mut links = HashSet::new();
    for s in &ir.stmts {
        if !matches!(s.ir, IR::SelectI) { continue; }
        let [upper_cond, lower, hi_id] = s.arguments.as_slice() else { continue };
        let Some(hi) = constant_int(ir, *hi_id) else { continue };
        let upper = &ir.stmts[*upper_cond as usize];
        if !matches!(upper.ir, IR::LtI) || upper.arguments != [*lower, *hi_id] { continue; }
        let lower_select = &ir.stmts[*lower as usize];
        if !matches!(lower_select.ir, IR::SelectI) { continue; }
        let [lower_cond, input, lo_id] = lower_select.arguments.as_slice() else { continue };
        let Some(lo) = constant_int(ir, *lo_id) else { continue };
        if lo > hi || hi as i128 - lo as i128 >= 1i128 << 32 { continue; }
        let comparison = &ir.stmts[*lower_cond as usize];
        if !matches!(comparison.ir, IR::GtI) || comparison.arguments != [*input, *lo_id] { continue; }
        if only_user(ir, uses, *upper_cond).map(|u| u.stmt_id) != Some(s.stmt_id)
            || only_user(ir, uses, *lower_cond).map(|u| u.stmt_id) != Some(*lower) { continue; }
        if !uses.get(lower).is_some_and(|users| users.len() == 2
            && users.contains(upper_cond) && users.contains(&s.stmt_id)) { continue; }
        if [*upper_cond, *lower, *lower_cond].iter().any(|id| roots.contains(&ir.stmts[*id as usize].value_id)) { continue; }
        clips.insert(s.stmt_id, Clip { input: ir.stmts[*input as usize].value_id, lo, hi });
        links.extend([*upper_cond, *lower, *lower_cond]);
    }
    (clips, links)
}

/// Exact conditional increments and negated truth conversions, with only the
/// final result observable. No floating-point product or bound is moved.
fn conditional_arithmetic(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> (
    HashMap<u32, (ValueId, ValueId)>, HashSet<u32>, HashSet<u32>, HashMap<u32, ValueId>) {
    let roots: HashSet<_> = ir.stmts.iter().filter_map(|s| ir.provenance.affine_of(s.value_id))
        .flat_map(|v| v.terms.iter().map(|(root, _)| *root)).collect();
    let mut adds = HashMap::new(); let mut links = HashSet::new();
    let mut casts = HashSet::new(); let mut nots = HashMap::new();
    for s in &ir.stmts {
        if matches!(s.ir, IR::SelectI | IR::SelectF) {
            let [_, t, f] = s.arguments.as_slice() else { continue };
            let sum = &ir.stmts[*t as usize];
            if !matches!(sum.ir, IR::AddI | IR::AddF) || roots.contains(&sum.value_id)
                || only_user(ir, uses, *t).map(|u| u.stmt_id) != Some(s.stmt_id) { continue; }
            let [a, b] = sum.arguments.as_slice() else { continue };
            let delta = if a == f { *b } else if b == f { *a } else { continue };
            adds.insert(s.stmt_id, (ir.stmts[*f as usize].value_id, ir.stmts[delta as usize].value_id));
            links.insert(*t);
        }
        if matches!(s.ir, IR::BoolCast) && !roots.contains(&s.value_id) {
            if let Some(user) = only_user(ir, uses, s.stmt_id).filter(|u| matches!(u.ir, IR::LogicalNot)) {
                if let [input] = s.arg_values.as_slice() {
                    casts.insert(s.stmt_id); nots.insert(user.stmt_id, *input);
                }
            }
        }
    }
    (adds, links, casts, nots)
}

/// Connected bitwise dataflow around conditional XOR updates. These are
/// generic integer bit recurrences, not arithmetic regions: retain Boolean
/// lanes between operations to avoid repeated decomposition/recomposition.
fn bitvector_islands(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> (HashSet<u32>, HashSet<u32>) {
    let eligible = |id: u32| matches!(ir.stmts[id as usize].ir,
        IR::BitAndI | IR::BitOrI | IR::BitXorI | IR::BitNotI | IR::ShrI | IR::ShlI | IR::SelectI);
    let mut pending = Vec::new();
    for s in &ir.stmts {
        if !matches!(s.ir, IR::SelectI) { continue; }
        let [_, t, f] = s.arguments.as_slice() else { continue };
        if [(*t, *f), (*f, *t)].iter().any(|(xor, other)| {
            let producer = &ir.stmts[*xor as usize];
            matches!(producer.ir, IR::BitXorI) && producer.arguments.contains(other)
        }) { pending.push(s.stmt_id); }
    }
    let mut nodes = HashSet::new();
    while let Some(id) = pending.pop() {
        if !eligible(id) || !nodes.insert(id) { continue; }
        pending.extend(ir.stmts[id as usize].arguments.iter().copied());
        if let Some(users) = uses.get(&id) { pending.extend(users.iter().copied()); }
    }
    let roots: HashSet<_> = ir.stmts.iter().filter_map(|s| ir.provenance.affine_of(s.value_id))
        .flat_map(|v| v.terms.iter().map(|(root, _)| *root)).collect();
    let links = nodes.iter().copied().filter(|id| !roots.contains(&ir.stmts[*id as usize].value_id)
        && uses.get(id).is_some_and(|users| !users.is_empty() && users.iter().all(|u| nodes.contains(u)
            && !(matches!(ir.stmts[*u as usize].ir, IR::SelectI) && ir.stmts[*u as usize].arguments[0] == *id))))
        .collect();
    (nodes, links)
}

/// Reserve provenance roots for address consumers. Ordinary arithmetic,
/// assertions, and public exposure read their explicit scalar arguments, not
/// the roots in those arguments' affine metadata. Their explicit uses already
/// prevent an interior node from being absorbed into an unrelated tree.
fn observed_affine_roots(ir: &IRGraph) -> HashSet<ValueId> {
    ir.stmts.iter().filter(|s| matches!(s.ir,
        IR::RomRead { .. } | IR::RamRead { .. } | IR::RamWrite { .. }
        | IR::SelectManyI { .. } | IR::SelectManyF { .. }))
        .flat_map(|s| s.arg_values.iter())
        .filter_map(|id| ir.provenance.affine_of(*id))
        .flat_map(|view| view.terms.iter().map(|(root, _)| *root)).collect()
}
