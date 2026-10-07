//! Memory groups: how an array that the program writes at runtime is built
//! (`docs/array-ssa.md`, section 4).
//!
//! A **version tree** is an `ArrayInit` with the versions `ArrayWrite`s make
//! from it (`crate::ir_array`). Its **memory group** (`OpKind::Versions`)
//! holds the tree and, as lanes, its reads at constant addresses; its reads
//! at a runtime position are lanes of element groups over the written
//! version's cells (`Group::version`). Reads of the `ArrayInit` itself are
//! element reads of known cells and do not belong here.
//!
//! Every construction below decides the representation of the whole tree,
//! with its reads and writes together:
//!
//! * [`Overlay`]: every version's cells as scalars, `cell' = select(hit ∧ p,
//!   x, cell)` per write; the reads are then read by any element
//!   construction (the overlay produces the cells they need, `CELLS`);
//! * [`ZeroWrites`]: the same cells when every write stores the constant 0
//!   through one runtime position (`a[rows, :] = 0`), as products of keep
//!   bits shared by the cells at one coordinate;
//! * [`RamTrace`] (halo2-base) and [`NativeRam`] (Noir): one memory, the
//!   init and every write and read an access of it in statement order
//!   (`IR::RamInit` / `RamWrite` / `RamRead`).

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use crate::builder::IRBuilder;
use crate::ir::IRStatement;
use crate::ir_array::{ArrayAccess, ArrayElem};
use crate::ir_defs::IR;
use crate::lower::gather::constructions::{Encodings, ENCODING};
use crate::lower::{
    Candidate, Construction, Context, EmitCx, Emitter, Group, GroupId, Lane, OpKind, Requirement,
    Target,
};
use crate::types::{StmtId, Value};

/// The shared value of a written version: its cells, by address
/// (`Rc<Vec<Value>>`), at `Requirement::SharedAt { at: version, .. }`.
pub const CELLS: &str = "memory: cells of a version";

/// The program-wide requirement of the RAM trace (halo2-base).
pub const RAM: Requirement = Requirement::Program { name: super::super::challenge::RAM_NAME };

/// One write of a tree.
#[derive(Debug, Clone)]
pub struct Write {
    pub stmt: StmtId,
    /// The version it writes into.
    pub parent: StmtId,
    pub access: ArrayAccess,
    pub positions: Vec<StmtId>,
    /// The positions' constant values (`None`: a runtime position).
    pub constants: Vec<Option<i64>>,
    pub value: StmtId,
    pub predicate: StmtId,
}

/// A read of a written version.
#[derive(Debug, Clone)]
pub struct Read {
    pub stmt: StmtId,
    pub version: StmtId,
    pub access: ArrayAccess,
    /// The runtime position (`None`: a constant address, `access.offset`).
    pub position: Option<StmtId>,
    /// The position's constant values, as [`Write::constants`].
    pub constants: Vec<Option<i64>>,
}

/// A version tree with at least one write and one read of a written version.
#[derive(Debug, Clone)]
pub struct Tree {
    /// The `ArrayInit`.
    pub root: StmtId,
    /// The root's cells.
    pub cells: Vec<StmtId>,
    pub elem: ArrayElem,
    /// The writes, in statement order.
    pub writes: Vec<Write>,
    /// The reads of written versions, in statement order.
    pub reads: Vec<Read>,
}

impl Tree {
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    fn write_of(&self, version: StmtId) -> Option<&Write> {
        self.writes.iter().find(|w| w.stmt == version)
    }

    /// The versions that a runtime read reads (their cells are wanted by the
    /// element groups reading them).
    pub fn read_versions(&self) -> BTreeSet<StmtId> {
        self.reads.iter().filter(|r| r.position.is_some()).map(|r| r.version).collect()
    }

    /// Whether the versions form one chain from the root, and every read
    /// reads the cell the newest version at its place in the program holds:
    /// it reads that version, or an older one that no later write before it
    /// can change at its cell (a fold skipped writes to other cells). The
    /// condition for one memory whose accesses are in statement order (the
    /// RAM discipline).
    pub fn is_chain_in_order(&self) -> bool {
        let mut newest = self.root;
        let mut next_write = self.writes.iter().peekable();
        let mut reads = self.reads.iter().peekable();
        // The writes of the chain so far, in order.
        let mut chain: Vec<&Write> = Vec::new();
        loop {
            // The next event in statement order.
            match (next_write.peek(), reads.peek()) {
                (Some(&w), Some(r)) if w.stmt < r.stmt => {
                    if w.parent != newest {
                        return false;
                    }
                    newest = w.stmt;
                    chain.push(w);
                    next_write.next();
                }
                (Some(&w), None) => {
                    if w.parent != newest {
                        return false;
                    }
                    newest = w.stmt;
                    chain.push(w);
                    next_write.next();
                }
                (_, Some(r)) => {
                    // The writes after the read's version on the chain.
                    let later = match chain.iter().position(|w| w.stmt == r.version) {
                        Some(k) => &chain[k + 1..],
                        None if r.version == self.root => &chain[..],
                        None => return false,
                    };
                    if !later.iter().all(|w| w.access.disjoint_from(&w.constants, &r.access, &r.constants)) {
                        return false;
                    }
                    reads.next();
                }
                (None, None) => return true,
            }
        }
    }
}

/// The value of `s` when it is an integer constant.
fn constant_of(stmts: &[IRStatement], s: StmtId) -> Option<i64> {
    match stmts[s as usize].ir {
        IR::ConstantInt { value } => Some(value),
        _ => None,
    }
}

/// The trees of `stmts`: every `ArrayInit` with a write and a read of a
/// written version, by root (statement order).
///
/// # Panics
/// On a read of a written version at two or more runtime positions (the
/// frontend reads such cells through constant reads and an element read).
pub fn trees(stmts: &[IRStatement]) -> BTreeMap<StmtId, Tree> {
    let mut root_of: HashMap<StmtId, StmtId> = HashMap::new();
    let mut trees: BTreeMap<StmtId, Tree> = BTreeMap::new();
    for stmt in stmts {
        match &stmt.ir {
            IR::ArrayInit { elem, .. } => {
                root_of.insert(stmt.stmt_id, stmt.stmt_id);
                trees.insert(
                    stmt.stmt_id,
                    Tree { root: stmt.stmt_id, cells: stmt.arguments.clone(), elem: *elem, writes: Vec::new(), reads: Vec::new() },
                );
            }
            IR::ArrayWrite(access) => {
                let parent = stmt.arguments[0];
                let root = root_of[&parent];
                root_of.insert(stmt.stmt_id, root);
                let rank = access.rank();
                let positions = stmt.arguments[1..1 + rank].to_vec();
                trees.get_mut(&root).expect("a root").writes.push(Write {
                    stmt: stmt.stmt_id,
                    parent,
                    access: access.clone(),
                    constants: positions.iter().map(|&p| constant_of(stmts, p)).collect(),
                    positions,
                    value: stmt.arguments[1 + rank],
                    predicate: stmt.arguments[2 + rank],
                });
            }
            IR::ArrayRead(access) => {
                let version = stmt.arguments[0];
                if matches!(stmts[version as usize].ir, IR::ArrayInit { .. }) {
                    continue;
                }
                assert!(
                    access.rank() <= 1,
                    "a read of a written array version at {} runtime positions ({})",
                    access.rank(),
                    access.describe()
                );
                let root = root_of[&version];
                let position = stmt.arguments.get(1).copied();
                trees.get_mut(&root).expect("a root").reads.push(Read {
                    stmt: stmt.stmt_id,
                    version,
                    access: access.clone(),
                    position,
                    constants: position.map(|p| constant_of(stmts, p)).into_iter().collect(),
                });
            }
            _ => {}
        }
    }
    trees.retain(|_, t| !t.writes.is_empty() && !t.reads.is_empty());
    trees
}

/// The memory family's constructions of memory groups, in tie-break order.
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![&Overlay, &ZeroWrites, &RamTrace, &NativeRam]
}

/// The memory group's own lanes: its reads at constant addresses.
fn constant_read_lanes(group: &Group) -> impl Iterator<Item = Requirement> + '_ {
    group.lanes.iter().map(|l| Requirement::Lane(l.stmt))
}

/// The lanes of the element groups that read the tree's written versions.
fn runtime_read_lanes(tree: &Tree) -> impl Iterator<Item = Requirement> + '_ {
    tree.reads.iter().filter(|r| r.position.is_some()).map(|r| Requirement::Lane(r.stmt))
}

/// The group of the `Wrap` placeholder whose position lane `position` is,
/// when it is one (its hits are the wrap's `ENCODING`).
fn wrap_group_of(cx: &Context, position: StmtId) -> Option<GroupId> {
    let group = cx.group_of_lane(position)?;
    let is_position = group.lanes.iter().any(|l| l.stmt == position && l.output == super::wrap::POSITION);
    (matches!(group.kind, OpKind::Wrap { .. }) && is_position).then_some(group.id)
}

// ── Overlay ─────────────────────────────────────────────────────────────

/// Every version's cells as scalars: a write at positions `i` (one per
/// axis) with value `x` under predicate `p` makes each cell `c` its access
/// can name `select(hit_c ∧ p, x, old_c)`, `hit_c` the conjunction of the
/// per-axis bits `[i_k == c_k]`; every other cell is the parent's.
///
/// **Preconditions.** The `Wrap` encodings of positions that are wrap
/// lanes. Every position lies in its axis on every path of an accepted
/// witness: the frontend checks or proves it (and guards it where the code
/// does not run), as for every array access (`crate::ir_array`).
///
/// **Postconditions.** The cells of every version a runtime read reads
/// (`CELLS`), which the element groups read with any element construction;
/// the reads at constant addresses (the cells themselves).
///
/// **Soundness and completeness.** No hint. With every position in its
/// axis, exactly the cell the access names has all its bits set, so the
/// new cells are the version NumPy's write makes when `p` holds, and the
/// parent's cells when it does not; each is a function of its inputs.
///
/// **Cost.** Per write and cell its access can name: a select of two
/// variable branches (one product when the value is the constant 0), and
/// the conjunctions of its bits and the predicate; per (position,
/// coordinate) not from a wrap's encoding an equality test.
pub struct Overlay;

impl Construction for Overlay {
    fn name(&self) -> &'static str {
        "memory: overlay of selects"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some(tree) = group.tree.clone().filter(|_| group.kind == OpKind::Versions) else { return Vec::new() };
        let reach: HashMap<StmtId, Vec<(u32, u32)>> = tree.writes.iter().map(|w| (w.stmt, reachable(cx, w))).collect();
        let (wraps, tested) = Hits::plan(cx, &tree.writes, &reach);
        let pre = Hits::preconditions(&wraps);
        let u = crate::lower::gather::cost::element_units(cx.target);
        let mut cost: u64 = 0;
        for w in &tree.writes {
            let conjuncts = w.positions.len().saturating_sub(1) + usize::from(!is_true(cx, w.predicate));
            let cells: u64 = reach[&w.stmt].iter().map(|&(lo, hi)| u64::from(hi - lo + 1)).product();
            // A select storing the constant 0 is one product (`x - c x`).
            let select = if is_zero(cx, w.value) { zero_select(cx.target) } else { u.select_var };
            cost += cells * (select + conjuncts as u64 * u.select_var);
        }
        cost += tested.len() as u64 * u.eq_const;

        let post = tree
            .read_versions()
            .into_iter()
            .map(|at| Requirement::SharedAt { at, name: CELLS })
            .chain(constant_read_lanes(group))
            .collect();
        let hits = Hits { wraps, bits: HashMap::new() };
        vec![Candidate { pre, post, cost, emitter: Box::new(OverlayEmitter { tree, hits, reach, cells: HashMap::new() }) }]
    }
}

/// The hit bits `[position == c]` of a tree's writes: the wrap's encoding
/// for a position that is a wrap lane, else equality tests, each once.
struct Hits {
    /// The `Wrap` group of each write position that is a wrap's position.
    wraps: HashMap<StmtId, GroupId>,
    bits: HashMap<(StmtId, u32), Value>,
}

impl Hits {
    /// The wraps the writes' positions are lanes of (their `ENCODING`s are
    /// preconditions), and the (position, coordinate) pairs `reach` names
    /// that need an equality test.
    fn plan(cx: &Context, writes: &[Write], reach: &HashMap<StmtId, Vec<(u32, u32)>>) -> (HashMap<StmtId, GroupId>, BTreeSet<(StmtId, u32)>) {
        let mut wraps = HashMap::new();
        let mut tested = BTreeSet::new();
        for w in writes {
            for (&x, &(lo, hi)) in w.positions.iter().zip(&reach[&w.stmt]) {
                match wrap_group_of(cx, x) {
                    Some(g) => {
                        wraps.insert(x, g);
                    }
                    None => tested.extend((lo..=hi).map(|c| (x, c))),
                }
            }
        }
        (wraps, tested)
    }

    fn preconditions(wraps: &HashMap<StmtId, GroupId>) -> Vec<Requirement> {
        let mut pre: Vec<Requirement> = wraps.values().map(|&g| Requirement::Shared { group: g, name: ENCODING }).collect();
        pre.sort();
        pre.dedup();
        pre
    }

    /// `[position == c]`.
    fn hit(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, position: StmtId, c: u32) -> Value {
        if let Some(bit) = self.bits.get(&(position, c)) {
            return bit.clone();
        }
        let bit = match self.wraps.get(&position) {
            Some(&g) => cx.shared::<Encodings>(b, g, ENCODING)[0].equals(b, i64::from(c)),
            None => {
                let p = cx.value(position);
                let k = b.ir_constant_int(i64::from(c));
                b.ir_equal_i(&p, &k)
            }
        };
        self.bits.insert((position, c), bit.clone());
        bit
    }
}

/// The coordinates each position of `w` can take, per axis (inclusive):
/// its axis, narrowed by the bounds the program proves for the position
/// (the range engine, over its definition). A cell outside them is never
/// written.
fn reachable(cx: &Context, w: &Write) -> Vec<(u32, u32)> {
    w.positions
        .iter()
        .zip(&w.access.axes)
        .map(|(&x, axis)| {
            let last = axis.len - 1;
            match cx.facts.bounds(x) {
                Some((lo, hi)) if lo <= i64::from(last) && hi >= 0 => {
                    (lo.max(0) as u32, hi.min(i64::from(last)) as u32)
                }
                _ => (0, last),
            }
        })
        .collect()
}

fn is_true(cx: &Context, s: StmtId) -> bool {
    matches!(cx.stmt(s).ir, IR::ConstantBool { value: true } | IR::ConstantInt { value: 1 })
}

struct OverlayEmitter {
    tree: Rc<Tree>,
    hits: Hits,
    /// The coordinates each write's positions can take (`reachable`).
    reach: HashMap<StmtId, Vec<(u32, u32)>>,
    /// The cells of each version built so far.
    cells: HashMap<StmtId, Rc<Vec<Value>>>,
}

impl OverlayEmitter {
    /// The cells of `version`, built from its parent's.
    fn cells_of(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, version: StmtId) -> Rc<Vec<Value>> {
        if let Some(cells) = self.cells.get(&version) {
            return cells.clone();
        }
        let cells = match self.tree.write_of(version).cloned() {
            None => Rc::new(cx.values(&self.tree.cells)),
            Some(w) => {
                let mut cells = (*self.cells_of(b, cx, w.parent)).clone();
                self.write(b, cx, &w, &mut cells);
                Rc::new(cells)
            }
        };
        self.cells.insert(version, cells.clone());
        cells
    }

    /// Apply the write `w` to `cells`: each cell its positions can reach.
    fn write(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, w: &Write, cells: &mut [Value]) {
        let value = cx.value(w.value);
        let predicate = cx.value(w.predicate);
        let always = predicate.bool_val().or_else(|| predicate.int_val().map(|v| v != 0)) == Some(true);
        let ranges = self.reach[&w.stmt].clone();
        let mut coords: Vec<u32> = ranges.iter().map(|&(lo, _)| lo).collect();
        loop {
            let address = w.access.offset as usize
                + coords.iter().zip(&w.access.axes).map(|(&c, a)| c as usize * a.stride as usize).sum::<usize>();
            let mut hit: Option<Value> = None;
            for (k, &c) in coords.iter().enumerate() {
                let bit = self.hits.hit(b, cx, w.positions[k], c);
                hit = Some(match hit {
                    None => bit,
                    Some(h) => b.ir_logical_and(&h, &bit),
                });
            }
            let mut cond = hit.unwrap_or_else(|| b.ir_constant_bool(true));
            if !always {
                cond = b.ir_logical_and(&cond, &predicate);
            }
            cells[address] = match self.tree.elem {
                ArrayElem::Float => b.ir_select_f(&cond, &value, &cells[address]),
                ArrayElem::Int => b.ir_select_i(&cond, &value, &cells[address]),
            };
            // The next reachable coordinates, row-major (the last axis
            // fastest); done after the last.
            let mut k = coords.len();
            loop {
                if k == 0 {
                    return;
                }
                k -= 1;
                if coords[k] < ranges[k].1 {
                    coords[k] += 1;
                    break;
                }
                coords[k] = ranges[k].0;
            }
        }
    }
}

impl Emitter for OverlayEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let read = self.tree.reads.iter().find(|r| r.stmt == lane.stmt).cloned().expect("a read of this tree");
        let cells = self.cells_of(b, cx, read.version);
        cells[read.access.offset as usize].clone()
    }

    fn shared_at(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, at: StmtId, _name: &'static str) -> Rc<dyn Any> {
        self.cells_of(b, cx, at)
    }
}

// ── Zero writes ─────────────────────────────────────────────────────────

/// Every write stores the constant 0 through one runtime position,
/// unconditionally (`a[rows, :] = 0` and `a[:, cols] = 0` with index arrays
/// write one index at a time): a cell ends as `x · Π_t (1 - [i_t == c_t])`
/// over the writes `t` that can reach it, `c_t` its coordinate on write
/// `t`'s axis. The factors of the writes along one axis (one stride) depend
/// on the cell only through its coordinate there, so their product, the
/// cell's *keep* bit for that axis, is built once per coordinate and shared
/// by every cell at it; a cell is its value times its keep bits.
///
/// **Applies** when every write of the tree stores the constant 0 through
/// one runtime position under a constant true predicate.
///
/// **Preconditions, postconditions, soundness.** As [`Overlay`]'s: no hint;
/// with every position in its axis, a cell's keep bits are all 1 exactly
/// when no write hits it, so it is 0 when some write hits it and its old
/// value otherwise, NumPy's result of the writes.
///
/// **Cost.** Per (axis, coordinate) `1 - h` of the first factor and one
/// product per further factor, per cell one product per keep bit; the hits
/// as the overlay's (halo2-base 4 cells per operation, Noir 1 gate).
pub struct ZeroWrites;

/// One product of a bit and a cell (`select(c, 0, x)`, `x · keep`):
/// halo2-base 4 cells, Noir 1 gate (`scatter::write`'s measurements).
fn zero_select(target: Target) -> u64 {
    match target {
        Target::Halo2Base => 4,
        Target::Noir => 1,
    }
}

fn is_zero(cx: &Context, s: StmtId) -> bool {
    match cx.stmt(s).ir {
        IR::ConstantInt { value } => value == 0,
        IR::ConstantFloat { value } => value == 0.0,
        _ => false,
    }
}

/// For each version, the writes from the root to it (in order).
fn write_paths(tree: &Tree) -> HashMap<StmtId, Vec<Write>> {
    let mut paths: HashMap<StmtId, Vec<Write>> = HashMap::from([(tree.root, Vec::new())]);
    for w in &tree.writes {
        let mut path = paths[&w.parent].clone();
        path.push(w.clone());
        paths.insert(w.stmt, path);
    }
    paths
}

/// The keep factors of the cell at `address` after `writes`: per axis
/// stride, in the order the axes are first written, the (position,
/// coordinate) pairs of the writes that reach it.
fn keep_factors(writes: &[Write], reach: &HashMap<StmtId, Vec<(u32, u32)>>, address: usize) -> Vec<Vec<(StmtId, u32)>> {
    let mut strides: Vec<u32> = Vec::new();
    let mut groups: Vec<Vec<(StmtId, u32)>> = Vec::new();
    for w in writes {
        let axis = &w.access.axes[0];
        let (lo, hi) = reach[&w.stmt][0];
        let Some(delta) = address.checked_sub(w.access.offset as usize) else { continue };
        let stride = axis.stride as usize;
        let c = if stride == 0 { 0 } else { delta / stride };
        if (stride == 0 && delta != 0) || (stride != 0 && delta % stride != 0) || c < lo as usize || c > hi as usize {
            continue;
        }
        let group = match strides.iter().position(|&s| s == axis.stride) {
            Some(g) => g,
            None => {
                strides.push(axis.stride);
                groups.push(Vec::new());
                groups.len() - 1
            }
        };
        groups[group].push((w.positions[0], c as u32));
    }
    groups
}

/// The cells a tree's reads need, as (version, address): the cell of each
/// read at a constant address, every cell of a version read at a runtime
/// position.
fn needed_cells(tree: &Tree) -> Vec<(StmtId, usize)> {
    let mut cells: BTreeSet<(StmtId, usize)> = BTreeSet::new();
    for r in &tree.reads {
        match r.position {
            None => {
                cells.insert((r.version, r.access.offset as usize));
            }
            Some(_) => cells.extend((0..tree.len()).map(|a| (r.version, a))),
        }
    }
    cells.into_iter().collect()
}

impl Construction for ZeroWrites {
    fn name(&self) -> &'static str {
        "memory: zero writes by keep products"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some(tree) = group.tree.clone().filter(|_| group.kind == OpKind::Versions) else { return Vec::new() };
        if !tree.writes.iter().all(|w| w.access.rank() == 1 && is_true(cx, w.predicate) && is_zero(cx, w.value)) {
            return Vec::new();
        }
        let reach: HashMap<StmtId, Vec<(u32, u32)>> = tree.writes.iter().map(|w| (w.stmt, reachable(cx, w))).collect();
        let (wraps, tested) = Hits::plan(cx, &tree.writes, &reach);
        let pre = Hits::preconditions(&wraps);
        // The products the reads need: per cell value (a cell with the same
        // factors in two versions is one value) one per keep bit, and the
        // keep chains, shared by every cell with the same factors.
        let paths = write_paths(&tree);
        let mut values: BTreeSet<(usize, Vec<Vec<(StmtId, u32)>>)> = BTreeSet::new();
        let mut keeps: BTreeSet<Vec<(StmtId, u32)>> = BTreeSet::new();
        let mut products: u64 = 0;
        for (version, address) in needed_cells(&tree) {
            let groups = keep_factors(&paths[&version], &reach, address);
            if !values.insert((address, groups.clone())) {
                continue;
            }
            for factors in groups.iter().filter(|f| !f.is_empty()) {
                products += 1;
                // The chain's `1 - h` of its first factor, then a product
                // per further factor (each prefix once).
                for k in 1..=factors.len() {
                    if keeps.insert(factors[..k].to_vec()) {
                        products += 1;
                    }
                }
            }
        }
        let u = crate::lower::gather::cost::element_units(cx.target);
        let cost = products * zero_select(cx.target) + tested.len() as u64 * u.eq_const;
        let post = tree
            .read_versions()
            .into_iter()
            .map(|at| Requirement::SharedAt { at, name: CELLS })
            .chain(constant_read_lanes(group))
            .collect();
        let hits = Hits { wraps, bits: HashMap::new() };
        vec![Candidate {
            pre,
            post,
            cost,
            emitter: Box::new(ZeroEmitter { tree, hits, reach, paths, keeps: HashMap::new(), values: HashMap::new() }),
        }]
    }
}

struct ZeroEmitter {
    tree: Rc<Tree>,
    hits: Hits,
    reach: HashMap<StmtId, Vec<(u32, u32)>>,
    paths: HashMap<StmtId, Vec<Write>>,
    /// The keep bit of each list of factors built so far.
    keeps: HashMap<Vec<(StmtId, u32)>, Value>,
    /// The value of each cell with its keep factors built so far.
    values: HashMap<(usize, Vec<Vec<(StmtId, u32)>>), Value>,
}

impl ZeroEmitter {
    /// `Π (1 - [position == c])` over `factors`: `not` of the first hit,
    /// then each further hit zeroes it.
    fn keep(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, factors: &[(StmtId, u32)]) -> Value {
        if let Some(keep) = self.keeps.get(factors) {
            return keep.clone();
        }
        let (position, c) = *factors.last().expect("a factor");
        let hit = self.hits.hit(b, cx, position, c);
        let keep = match factors.len() {
            1 => b.ir_logical_not(&hit),
            k => {
                let rest = self.keep(b, cx, &factors[..k - 1]);
                let zero = b.ir_constant_int(0);
                b.ir_select_i(&hit, &zero, &rest)
            }
        };
        self.keeps.insert(factors.to_vec(), keep.clone());
        keep
    }

    /// The cell at `address` of `version`: the root's cell times its keep
    /// bits.
    fn cell(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, version: StmtId, address: usize) -> Value {
        let groups = keep_factors(&self.paths[&version], &self.reach, address);
        let key = (address, groups);
        if let Some(value) = self.values.get(&key) {
            return value.clone();
        }
        let dtype = match self.tree.elem {
            ArrayElem::Float => crate::types::NumberType::Float,
            ArrayElem::Int => crate::types::NumberType::Integer,
        };
        let mut value = cx.value(self.tree.cells[address]);
        for factors in key.1.iter().filter(|f| !f.is_empty()) {
            let keep = self.keep(b, cx, factors);
            value = crate::lower::scatter::arith::bit_times(b, &keep, &value, dtype);
        }
        self.values.insert(key, value.clone());
        value
    }
}

impl Emitter for ZeroEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let read = self.tree.reads.iter().find(|r| r.stmt == lane.stmt).cloned().expect("a read of this tree");
        self.cell(b, cx, read.version, read.access.offset as usize)
    }

    fn shared_at(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, at: StmtId, _name: &'static str) -> Rc<dyn Any> {
        let cells: Vec<Value> = (0..self.tree.len()).map(|a| self.cell(b, cx, at, a)).collect();
        Rc::new(cells)
    }
}

// ── RAM ─────────────────────────────────────────────────────────────────

/// One memory for the tree: `RamInit` of the root's cells, then each write
/// and read an access in statement order (`IR::RamWrite`, `IR::RamRead`) at
/// the flat address `offset + Σ i_k stride_k`; a predicated write is a read
/// of the old cell, a select and a write. halo2-base: rows of the program's
/// shared RAM trace (`prove/halo2_base/circuit/ram.rs`).
///
/// **Applies** when the versions form a chain whose reads all read the
/// newest version (`Tree::is_chain_in_order`), on halo2-base.
///
/// **Preconditions.** The RAM trace (`RAM`, paid once). The trace proves
/// the flat address in the program's memory space; that every position
/// lies in its axis (so the access stays in this array's addresses) is the
/// frontend's guarantee for every array access (its check or proof, and
/// the guard where the code does not run; `crate::ir_array`).
///
/// **Postconditions.** Every read of the tree.
///
/// **Soundness.** `circuit/ram.rs`: the trace pins every read to the last
/// earlier write at its address, or the root's cell; the accesses are in
/// statement order, which the chain condition makes the program's order.
///
/// **Cost** (halo2-base): about 8 cells per initial row and per access
/// (`layer-memory.md` 1.2), a select per predicated write.
pub struct RamTrace;

/// The same memory as a mutable native array on Noir (bb's RAM block), the
/// index of every access checked by `array_index`.
///
/// **Cost** (Noir): measured with private indices (`layer-memory.md` 1.2):
/// about `3.5 n` gates per access over `n` cells and 3 per cell.
pub struct NativeRam;

/// Cells per RAM access and per initial row (halo2-base).
const RAM_ACCESS_CELLS: u64 = 8;

impl Construction for RamTrace {
    fn name(&self) -> &'static str {
        "memory: RAM trace"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if cx.target != Target::Halo2Base {
            return Vec::new();
        }
        ram_candidate(group, cx, vec![RAM], |tree, cx| {
            let predicated = tree.writes.iter().filter(|w| !is_true(cx, w.predicate)).count() as u64;
            let accesses = tree.writes.len() as u64 + predicated + tree.reads.len() as u64;
            let u = crate::lower::gather::cost::element_units(cx.target);
            RAM_ACCESS_CELLS * (tree.len() as u64 + accesses) + predicated * u.select_var
        })
    }
}

impl Construction for NativeRam {
    fn name(&self) -> &'static str {
        "memory: native RAM"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if cx.target != Target::Noir {
            return Vec::new();
        }
        ram_candidate(group, cx, Vec::new(), |tree, cx| {
            let predicated = tree.writes.iter().filter(|w| !is_true(cx, w.predicate)).count() as u64;
            let accesses = tree.writes.len() as u64 + predicated + tree.reads.len() as u64;
            let n = tree.len() as u64;
            3 * n + (7 * n).div_ceil(2) * accesses + predicated * 2
        })
    }
}

fn ram_candidate(group: &Group, cx: &Context, pre: Vec<Requirement>, cost: impl Fn(&Tree, &Context) -> u64) -> Vec<Candidate> {
    let Some(tree) = group.tree.clone().filter(|_| group.kind == OpKind::Versions) else { return Vec::new() };
    if !tree.is_chain_in_order() {
        return Vec::new();
    }
    let post = constant_read_lanes(group).chain(runtime_read_lanes(&tree)).collect();
    let cost = cost(&tree, cx);
    vec![Candidate { pre, post, cost, emitter: Box::new(RamEmitter { tree, memory: None, written: 0, last_read: None }) }]
}

struct RamEmitter {
    tree: Rc<Tree>,
    /// The memory's current version.
    memory: Option<Value>,
    /// The writes emitted so far (a prefix of the chain).
    written: usize,
    /// The last read emitted.
    last_read: Option<StmtId>,
}

impl RamEmitter {
    /// `offset + Σ i_k stride_k`.
    fn address(b: &mut IRBuilder, cx: &EmitCx, access: &ArrayAccess, positions: &[StmtId]) -> Value {
        let mut address = b.ir_constant_int(i64::from(access.offset));
        for (axis, &p) in access.axes.iter().zip(positions) {
            let p = cx.value(p);
            let scaled = if axis.stride == 1 {
                p
            } else {
                let stride = b.ir_constant_int(i64::from(axis.stride));
                b.ir_mul_i(&p, &stride)
            };
            address = b.ir_add_i(&address, &scaled);
        }
        address
    }

    /// Emit the writes of the chain up to `version`, unless the memory is
    /// there or past it (a read of an older version whose cell the later
    /// writes do not change, `Tree::is_chain_in_order`).
    fn advance(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, version: StmtId) {
        let (len, elem) = (self.tree.len() as u32, self.tree.elem);
        let done = |e: &Self| version == e.tree.root || e.tree.writes[..e.written].iter().any(|w| w.stmt == version);
        while !done(self) {
            let w = self.tree.writes[self.written].clone();
            let memory = self.memory.clone().expect("begin ran");
            let address = Self::address(b, cx, &w.access, &w.positions);
            let value = cx.value(w.value);
            let predicate = cx.value(w.predicate);
            let stored = if predicate.bool_val().or_else(|| predicate.int_val().map(|v| v != 0)) == Some(true) {
                value
            } else {
                let old = b.ir_ram_read(&memory, len, elem, &address);
                match elem {
                    ArrayElem::Float => b.ir_select_f(&predicate, &value, &old),
                    ArrayElem::Int => b.ir_select_i(&predicate, &value, &old),
                }
            };
            self.memory = Some(b.ir_ram_write(&memory, len, elem, &address, &stored));
            self.written += 1;
        }
    }
}

impl Emitter for RamEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let cells = cx.values(&self.tree.cells);
        self.memory = Some(b.ir_ram_init(&cells, self.tree.elem));
    }

    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let read = self.tree.reads.iter().find(|r| r.stmt == lane.stmt).cloned().expect("a read of this tree");
        // Every read is served by the current memory: the reads must come in
        // statement order (`Tree::is_chain_in_order` justifies the memory's
        // state per position in that order), and the writes the memory is
        // past the read's version must not change the read's cell.
        assert!(
            self.last_read.is_none_or(|last| last < read.stmt),
            "RAM lowering: read {} emitted after read {:?}",
            read.stmt,
            self.last_read
        );
        self.last_read = Some(read.stmt);
        self.advance(b, cx, read.version);
        let after = match self.tree.writes[..self.written].iter().position(|w| w.stmt == read.version) {
            Some(k) => k + 1,
            None => {
                assert_eq!(read.version, self.tree.root, "RAM lowering: a read of an unwritten version");
                0
            }
        };
        assert!(
            self.tree.writes[after..self.written].iter().all(|w| w.access.disjoint_from(&w.constants, &read.access, &read.constants)),
            "RAM lowering: read {} of version {} after a write that may change its cell",
            read.stmt,
            read.version
        );
        let positions: Vec<StmtId> = read.position.into_iter().collect();
        let address = Self::address(b, cx, &read.access, &positions);
        let memory = self.memory.clone().expect("begin ran");
        b.ir_ram_read(&memory, self.tree.len() as u32, self.tree.elem, &address)
    }
}

/// What an element group over a written version's cells needs from the
/// version's memory group (its cells, `CELLS`).
pub fn cells_requirement(version: StmtId) -> Requirement {
    Requirement::SharedAt { at: version, name: CELLS }
}
