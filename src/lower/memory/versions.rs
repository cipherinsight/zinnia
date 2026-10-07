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
    Bound, Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, GroupId, Lane, OpKind, Requirement,
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
    /// reads the newest version at its place in the program (between its
    /// version's write and the next one): the condition for one memory whose
    /// accesses are in statement order (the RAM discipline).
    pub fn is_chain_in_order(&self) -> bool {
        let mut newest = self.root;
        let mut next_write = self.writes.iter().peekable();
        let mut reads = self.reads.iter().peekable();
        loop {
            // The next event in statement order.
            match (next_write.peek(), reads.peek()) {
                (Some(w), Some(r)) if w.stmt < r.stmt => {
                    if w.parent != newest {
                        return false;
                    }
                    newest = w.stmt;
                    next_write.next();
                }
                (Some(w), None) => {
                    if w.parent != newest {
                        return false;
                    }
                    newest = w.stmt;
                    next_write.next();
                }
                (_, Some(r)) => {
                    if r.version != newest {
                        return false;
                    }
                    reads.next();
                }
                (None, None) => return true,
            }
        }
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
                trees.get_mut(&root).expect("a root").writes.push(Write {
                    stmt: stmt.stmt_id,
                    parent,
                    access: access.clone(),
                    positions: stmt.arguments[1..1 + rank].to_vec(),
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
                trees.get_mut(&root).expect("a root").reads.push(Read {
                    stmt: stmt.stmt_id,
                    version,
                    access: access.clone(),
                    position: stmt.arguments.get(1).copied(),
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
    vec![&Overlay, &RamTrace, &NativeRam]
}

/// `0 <= p < len` for every position of a write or a read of the tree (the
/// operations' precondition: the program's index checks).
fn position_facts(tree: &Tree) -> Vec<Requirement> {
    let writes = tree.writes.iter().flat_map(|w| w.positions.iter().zip(&w.access.axes));
    let reads = tree.reads.iter().filter_map(|r| Some((r.position.as_ref()?, r.access.axes.first()?)));
    let mut facts: Vec<Requirement> = writes
        .chain(reads)
        .map(|(&x, axis)| {
            Requirement::Fact(Fact::everywhere(Claim::InRange { x, lo: 0, hi: Bound::Const(i64::from(axis.len)) }))
        })
        .collect();
    facts.sort();
    facts.dedup();
    facts
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
/// **Preconditions.** `0 <= i_k < len_k` for every write's position (the
/// program's checks); the `Wrap` encodings of positions that are wrap lanes.
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
/// variable branches, and the conjunctions of its bits and the predicate;
/// per (position, coordinate) not from a wrap's encoding an equality test.
pub struct Overlay;

impl Construction for Overlay {
    fn name(&self) -> &'static str {
        "memory: overlay of selects"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some(tree) = group.tree.clone().filter(|_| group.kind == OpKind::Versions) else { return Vec::new() };
        let mut pre = position_facts(&tree);
        let mut tested: BTreeSet<(StmtId, u32)> = BTreeSet::new();
        let mut wraps: HashMap<StmtId, GroupId> = HashMap::new();
        let mut terms: u64 = 0;
        for w in &tree.writes {
            for (&x, axis) in w.positions.iter().zip(&w.access.axes) {
                match wrap_group_of(cx, x) {
                    Some(g) => {
                        pre.push(Requirement::Shared { group: g, name: ENCODING });
                        wraps.insert(x, g);
                    }
                    None => tested.extend((0..axis.len).map(|c| (x, c))),
                }
            }
            let conjuncts = w.positions.len().saturating_sub(1) + usize::from(!is_true(cx, w.predicate));
            terms += w.access.line().len() as u64 * (1 + conjuncts as u64);
        }
        pre.sort();
        pre.dedup();
        let u = crate::lower::gather::cost::element_units(cx.target);
        let cost = terms * u.select_var + tested.len() as u64 * u.eq_const;
        let post = tree
            .read_versions()
            .into_iter()
            .map(|at| Requirement::SharedAt { at, name: CELLS })
            .chain(constant_read_lanes(group))
            .collect();
        vec![Candidate { pre, post, cost, emitter: Box::new(OverlayEmitter { tree, wraps, cells: HashMap::new(), hits: HashMap::new() }) }]
    }
}

fn is_true(cx: &Context, s: StmtId) -> bool {
    matches!(cx.stmt(s).ir, IR::ConstantBool { value: true } | IR::ConstantInt { value: 1 })
}

struct OverlayEmitter {
    tree: Rc<Tree>,
    /// The `Wrap` group of each write position that is a wrap's position.
    wraps: HashMap<StmtId, GroupId>,
    /// The cells of each version built so far.
    cells: HashMap<StmtId, Rc<Vec<Value>>>,
    /// `[position == coordinate]` bits.
    hits: HashMap<(StmtId, u32), Value>,
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

    /// Apply the write `w` to `cells`.
    fn write(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, w: &Write, cells: &mut [Value]) {
        let value = cx.value(w.value);
        let predicate = cx.value(w.predicate);
        let always = predicate.bool_val().or_else(|| predicate.int_val().map(|v| v != 0)) == Some(true);
        let mut coords = vec![0u32; w.access.rank()];
        for address in w.access.line() {
            let mut hit: Option<Value> = None;
            for (k, &c) in coords.iter().enumerate() {
                let bit = self.hit(b, cx, w.positions[k], c);
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
            // Row-major order of the positions (the last axis fastest),
            // matching `ArrayAccess::line`.
            for k in (0..coords.len()).rev() {
                coords[k] += 1;
                if coords[k] < w.access.axes[k].len {
                    break;
                }
                coords[k] = 0;
            }
        }
    }

    /// `[position == c]`: the wrap's hit bit when the position is a wrap
    /// lane, else an equality test (once per pair).
    fn hit(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, position: StmtId, c: u32) -> Value {
        if let Some(bit) = self.hits.get(&(position, c)) {
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
        self.hits.insert((position, c), bit.clone());
        bit
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
/// **Preconditions.** Every position in its axis (the program's checks:
/// the trace proves the flat address in the program's memory space, the
/// checks keep it in this array's); the RAM trace (`RAM`, paid once).
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

fn ram_candidate(group: &Group, cx: &Context, mut pre: Vec<Requirement>, cost: impl Fn(&Tree, &Context) -> u64) -> Vec<Candidate> {
    let Some(tree) = group.tree.clone().filter(|_| group.kind == OpKind::Versions) else { return Vec::new() };
    if !tree.is_chain_in_order() {
        return Vec::new();
    }
    pre.extend(position_facts(&tree));
    let post = constant_read_lanes(group).chain(runtime_read_lanes(&tree)).collect();
    let cost = cost(&tree, cx);
    vec![Candidate { pre, post, cost, emitter: Box::new(RamEmitter { tree, memory: None, written: 0 }) }]
}

struct RamEmitter {
    tree: Rc<Tree>,
    /// The memory's current version.
    memory: Option<Value>,
    /// The writes emitted so far (a prefix of the chain).
    written: usize,
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

    /// Emit the writes of the chain up to `version`.
    fn advance(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, version: StmtId) {
        let (len, elem) = (self.tree.len() as u32, self.tree.elem);
        while self.current() != version {
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

    fn current(&self) -> StmtId {
        if self.written == 0 {
            self.tree.root
        } else {
            self.tree.writes[self.written - 1].stmt
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
        self.advance(b, cx, read.version);
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
