//! The gather family's constructions over `IR::ArrayOp` placeholders.
//!
//! Operators (`OpKind`): `FirstSet`, `LastSet`, `ArgmaxOfFlags` (a position
//! of a flag vector), `PermutationScatter` (`c[p] = v` for a permutation
//! `p`), `Window` (a slice with runtime bounds) and `Element` (a read at a
//! runtime index checked into range). Each construction below names the
//! generator it emits and its cost formula (`cost.rs`); the generators' doc
//! comments carry the preconditions, postconditions and soundness
//! arguments.
//!
//! A scan by running products, and a permutation scatter of `arange`, also
//! produce the shared value [`ENCODING`] (each lane's one-hot, and a scan's
//! thermometer). A window or an element read whose index is such a lane
//! (plus a constant) offers a second candidate that consumes it instead of
//! building indicators by equality tests and comparisons; the selection
//! weighs the pair against the cheaper scan (a select chain) with the
//! self-contained window or read.

use std::any::Any;
use std::rc::Rc;

use crate::builder::IRBuilder;
use crate::ir_defs::IR;
use crate::lower::{Candidate, Construction, Context, EmitCx, Emitter, Group, GroupId, Lane, OpKind, Requirement};
use crate::types::{NumberType, StmtId, Value};

use super::cost::{self, AxisCost, MaskCost};
use super::encoding::IndexEncoding;
use super::scan::{self, ScanKind};
use super::vanishing::forced_zero_by_unset;
use super::window::{window_shift, AxisWindow};

/// The shared value of a scan or permutation-scatter group: the
/// `IndexEncoding` of each lane, by position (`Vec<IndexEncoding>`).
pub const ENCODING: &str = "encoding";

/// The encodings of a group's lanes, by position.
type Encodings = Vec<IndexEncoding>;

/// The family's constructions, in the order the selection breaks ties.
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![
        &ScanByRunningProducts,
        &ScanBySelectChain,
        &ScatterByHits,
        &WindowShift,
        &ElementOfWrite,
        &ElementByOneHot,
        &ElementBySelectChain,
        &ElementByIndicators,
        &ElementByMemory,
    ]
}

fn lanes(group: &Group) -> impl Iterator<Item = Requirement> + '_ {
    group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt))
}

// ── Scans ─────────────────────────────────────────────────────────────

/// `scan::scan`: the position by running products, with its encoding as the
/// shared value [`ENCODING`]. Cost `cost::scan_running_products`.
pub struct ScanByRunningProducts;

/// `scan::select_chain`: the position by a select chain, no encoding. Cost
/// `cost::scan_select_chain`.
pub struct ScanBySelectChain;

impl Construction for ScanByRunningProducts {
    fn name(&self) -> &'static str {
        "gather: scan by running products"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some(kind) = ScanKind::of(group.kind) else { return Vec::new() };
        vec![Candidate {
            pre: Vec::new(),
            post: lanes(group).chain([Requirement::Shared { group: group.id, name: ENCODING }]).collect(),
            cost: cost::scan_running_products(group.shared.len() as u64, kind == ScanKind::Argmax, cx.target),
            emitter: Box::new(ScanEmitter { kind, flags: group.shared.clone(), by_products: true, result: None }),
        }]
    }
}

impl Construction for ScanBySelectChain {
    fn name(&self) -> &'static str {
        "gather: scan by a select chain"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some(kind) = ScanKind::of(group.kind) else { return Vec::new() };
        vec![Candidate {
            pre: Vec::new(),
            post: lanes(group).collect(),
            cost: cost::scan_select_chain(group.shared.len() as u64, cx.target),
            emitter: Box::new(ScanEmitter { kind, flags: group.shared.clone(), by_products: false, result: None }),
        }]
    }
}

struct ScanEmitter {
    kind: ScanKind,
    flags: Vec<StmtId>,
    by_products: bool,
    result: Option<(Value, Option<IndexEncoding>)>,
}

impl Emitter for ScanEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let flags = cx.values(&self.flags);
        self.result = Some(if self.by_products {
            let (value, encoding) = scan::scan(b, self.kind, &flags);
            (value, Some(encoding))
        } else {
            (scan::select_chain(b, self.kind, &flags), None)
        });
    }

    fn lane(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _lane: &Lane) -> Value {
        self.result.as_ref().expect("started").0.clone()
    }

    fn shared(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _group: GroupId, _name: &'static str) -> Rc<dyn Any> {
        let encoding = self.result.as_ref().and_then(|r| r.1.clone()).expect("a scan by running products");
        Rc::new(vec![encoding] as Encodings)
    }
}

/// Where the one-hot of `stmt` comes from: `(group, position, c)` when
/// `stmt` is, plus the constant `c` (through `+`/`-` of constants), lane
/// `position` of a group whose producer may offer `ENCODING` (a scan, or a
/// permutation scatter of `k + c'`).
fn encoding_source(cx: &Context, stmt: StmtId) -> Option<(GroupId, usize, i64)> {
    let (root, offset) = super::strip_constant_offset(&cx.graph.stmts, stmt);
    let group = cx.group_of_lane(root)?;
    let encoded = ScanKind::of(group.kind).is_some() || scatter_offset(cx, group).is_some();
    let position = group.lanes.iter().find(|l| l.stmt == root)?.position? as usize;
    encoded.then_some((group.id, position, offset))
}

/// `c` when `group` is a permutation scatter of the values `k + c`.
fn scatter_offset(cx: &Context, group: &Group) -> Option<i64> {
    if group.kind != OpKind::PermutationScatter {
        return None;
    }
    let values: Vec<Option<i64>> = group.shared[group.shared.len() / 2..]
        .iter()
        .map(|&v| match cx.stmt(v).ir {
            IR::ConstantInt { value } => Some(value),
            _ => None,
        })
        .collect();
    super::encoded_values(&values)
}

/// The encoding of `(group, position, c)`'s value (`encoding_source`).
fn encoding_at(b: &mut IRBuilder, cx: &mut EmitCx, (group, position, c): (GroupId, usize, i64)) -> IndexEncoding {
    cx.shared::<Encodings>(b, group, ENCODING)[position].shifted(c)
}

// ── Permutation scatters ──────────────────────────────────────────────

/// `c[j] = sum_k [p[k] == j] v[k]` (`scatter_by_hits`): exactly one hit
/// per `j` since `p` is a permutation (the precondition the frontend
/// checked). The hits are the precondition's own (`emit_is_permutation`;
/// deduplicated after lowering). When `v[k] = k + c'`, the hits are each
/// cell's one-hot, offered as `ENCODING`. Cost: `n^2` equality tests and
/// multiply-adds (`cost::permutation_scatter`).
pub struct ScatterByHits;

impl Construction for ScatterByHits {
    fn name(&self) -> &'static str {
        "gather: permutation scatter by hits"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::PermutationScatter {
            return Vec::new();
        }
        let n = group.shared.len() / 2;
        let offset = scatter_offset(cx, group);
        let mut post: Vec<Requirement> = lanes(group).collect();
        if offset.is_some() {
            post.push(Requirement::Shared { group: group.id, name: ENCODING });
        }
        vec![Candidate {
            pre: Vec::new(),
            post,
            cost: cost::permutation_scatter(n as u64, cx.target),
            emitter: Box::new(ScatterEmitter { operands: group.shared.clone(), offset, hits: Vec::new(), cells: Vec::new() }),
        }]
    }
}

struct ScatterEmitter {
    operands: Vec<StmtId>,
    offset: Option<i64>,
    /// `hits[j][k] = [p[k] == j]`.
    hits: Vec<Vec<Value>>,
    cells: Vec<Value>,
}

impl Emitter for ScatterEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let values = cx.values(&self.operands);
        let n = values.len() / 2;
        let (p, v) = values.split_at(n);
        let dtype = dtype_of(v);
        self.hits = (0..n)
            .map(|j| {
                let target = b.ir_constant_int(j as i64);
                p.iter().map(|pk| b.ir_equal_i(pk, &target)).collect()
            })
            .collect();
        self.cells = self
            .hits
            .iter()
            .map(|hits| {
                let terms = hits.iter().zip(v).map(|(h, vk)| super::window::scale_by_bit(b, h, vk, dtype)).collect();
                super::window::sum_typed(b, terms, dtype)
            })
            .collect();
    }

    fn lane(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, lane: &Lane) -> Value {
        self.cells[lane.position.expect("a scatter lane has a position") as usize].clone()
    }

    fn shared(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _group: GroupId, _name: &'static str) -> Rc<dyn Any> {
        let lo = self.offset.expect("values k + c");
        let encodings: Encodings =
            self.hits.iter().map(|hits| IndexEncoding { lo, onehot: hits.clone(), below: None }).collect();
        Rc::new(encodings)
    }
}

// ── Windows ───────────────────────────────────────────────────────────

/// `window::window_shift`. Two candidates: selectors from equality tests
/// and comparisons only, and (when a start or stop is a scan's result plus a
/// constant) selectors read off the scans' encodings, with the padding mask
/// dropped where `vanishing` shows the padding is 0. Cost
/// `cost::window_shift`.
pub struct WindowShift;

/// One axis of a window group: its length and bound statements.
struct WindowAxis {
    n: usize,
    start: StmtId,
    stop: StmtId,
}

/// How an axis's selectors are built in one candidate.
#[derive(Clone)]
struct AxisPlan {
    start_from: Option<(GroupId, usize, i64)>,
    stop_from: Option<(GroupId, usize, i64)>,
    padding_zero: bool,
}

impl Construction for WindowShift {
    fn name(&self) -> &'static str {
        "gather: window shift"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let OpKind::Window { rank, dims } = group.kind else { return Vec::new() };
        let shape: Vec<usize> = dims[..rank as usize].iter().map(|&d| d as usize).collect();
        let cells = group.shared.len() - 2 * shape.len();
        let axes: Vec<WindowAxis> = (0..shape.len())
            .map(|a| WindowAxis { n: shape[a], start: group.shared[cells + 2 * a], stop: group.shared[cells + 2 * a + 1] })
            .collect();
        let plain: Vec<AxisPlan> =
            axes.iter().map(|_| AxisPlan { start_from: None, stop_from: None, padding_zero: false }).collect();
        let encoded: Vec<AxisPlan> = axes
            .iter()
            .enumerate()
            .map(|(a, axis)| {
                let start_from = encoding_source(cx, axis.start);
                let stop_from = encoding_source(cx, axis.stop)
                    .filter(|(g, _, _)| ScanKind::of(cx.groups[*g].kind).is_some_and(ScanKind::has_thermometer));
                let padding_zero = stop_from.is_some_and(|(g, _, c)| {
                    padding_is_zero(cx, &cx.groups[g], c, &group.shared[..cells], &shape, a)
                });
                AxisPlan { start_from, stop_from, padding_zero }
            })
            .collect();
        let mut out = vec![window_candidate(group, cx, &shape, &axes, plain)];
        if encoded.iter().any(|p| p.start_from.is_some() || p.stop_from.is_some()) {
            out.push(window_candidate(group, cx, &shape, &axes, encoded));
        }
        out
    }
}

/// Whether every source cell at a position past `stop = last + c` of axis
/// `axis` is 0: `c >= 1`, the scan is a `LastSet` over one flag per
/// position, and each flag being 0 forces the cells at its position to 0.
fn padding_is_zero(cx: &Context, scan: &Group, c: i64, cells: &[StmtId], shape: &[usize], axis: usize) -> bool {
    if scan.kind != OpKind::LastSet || c < 1 || scan.shared.len() != shape[axis] {
        return false;
    }
    let strides = crate::helpers::shape_arith::row_major_strides(shape);
    scan.shared.iter().enumerate().all(|(k, &flag)| {
        let zeros = forced_zero_by_unset(&cx.graph.stmts, flag);
        cells.iter().enumerate().all(|(flat, cell)| (flat / strides[axis]) % shape[axis] != k || zeros.contains(cell))
    })
}

fn window_candidate(group: &Group, cx: &Context, shape: &[usize], axes: &[WindowAxis], plans: Vec<AxisPlan>) -> Candidate {
    let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    let axis_costs: Vec<AxisCost> = axes
        .iter()
        .zip(&plans)
        .map(|(axis, plan)| AxisCost {
            n: axis.n as u64,
            start_free: plan.start_from.is_some(),
            mask: if plan.padding_zero || stop_reaches_end(cx, axis) {
                MaskCost::None
            } else if plan.stop_from.is_some() {
                MaskCost::Thermometer
            } else {
                MaskCost::Compared
            },
        })
        .collect();
    let pre = plans
        .iter()
        .flat_map(|p| [p.start_from, p.stop_from])
        .flatten()
        .map(|(g, _, _)| Requirement::Shared { group: g, name: ENCODING })
        .collect();
    Candidate {
        pre,
        post: lanes(group).collect(),
        cost: cost::window_shift(&dims, &axis_costs, cx.target),
        emitter: Box::new(WindowEmitter {
            shape: shape.to_vec(),
            operands: group.shared.clone(),
            plans,
            cells: Vec::new(),
        }),
    }
}

/// `stop` is a constant `>= n`.
fn stop_reaches_end(cx: &Context, axis: &WindowAxis) -> bool {
    matches!(cx.stmt(axis.stop).ir, IR::ConstantInt { value } if value >= axis.n as i64)
}

struct WindowEmitter {
    shape: Vec<usize>,
    operands: Vec<StmtId>,
    plans: Vec<AxisPlan>,
    cells: Vec<Value>,
}

impl Emitter for WindowEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let values = cx.values(&self.operands);
        let cells = values.len() - 2 * self.shape.len();
        let dtype = if values[..cells].iter().any(|v| matches!(v, Value::Float(_))) {
            NumberType::Float
        } else {
            NumberType::Integer
        };
        let mut windows = Vec::with_capacity(self.shape.len());
        for (axis, plan) in self.plans.iter().enumerate() {
            let encoding =
                |cx: &mut EmitCx, b: &mut IRBuilder, from: Option<(GroupId, usize, i64)>| from.map(|f| encoding_at(b, cx, f));
            windows.push(AxisWindow {
                axis,
                start: values[cells + 2 * axis].clone(),
                stop: values[cells + 2 * axis + 1].clone(),
                start_encoding: encoding(cx, b, plan.start_from),
                stop_encoding: encoding(cx, b, plan.stop_from),
                padding_zero: plan.padding_zero,
            });
        }
        self.cells = window_shift(b, values[..cells].to_vec(), &self.shape, &windows, dtype);
    }

    fn lane(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, lane: &Lane) -> Value {
        self.cells[lane.position.expect("a window lane has a position") as usize].clone()
    }
}

// ── Element reads ─────────────────────────────────────────────────────

/// `element::gather_by_onehot` for one read whose index has a one-hot
/// (`encoding_source`), with that encoding. Cost `cost::element_by_onehot`.
pub struct ElementByOneHot;

/// A read at index `i` of cells that a write at the same `i` produced: every
/// cell `k` is `select([i == k], v, old_k)` with the same statements `i` and
/// `v` (the frontend's one-hot write `a[..., i, ...] = v`). The read is `v`.
///
/// **Precondition.** `0 <= i < n` where the read runs (the element
/// placeholder's).
///
/// **Soundness.** No constraint: cell `i` is `select([i == i], v, old_i)`,
/// which is `v` on every backend (an exact equality bit), so `x[i] = v`.
/// The cells stay as they are for any other reader.
///
/// **Cost.** 0 (the selects are kept only if something else reads them).
pub struct ElementOfWrite;

impl Construction for ElementOfWrite {
    fn name(&self) -> &'static str {
        "gather: element of a write at the same index"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::Element {
            return Vec::new();
        }
        let written: Option<Vec<(StmtId, StmtId)>> =
            group.lanes.iter().map(|lane| Some((lane.stmt, written_at(cx, &group.shared, lane.own[0])?))).collect();
        let Some(written) = written else { return Vec::new() };
        vec![Candidate {
            pre: Vec::new(),
            post: lanes(group).collect(),
            cost: 0,
            emitter: Box::new(WrittenReadEmitter { written }),
        }]
    }
}

/// `v` when every cell `k` of `cells` is `select([index == k], v, _)` for one
/// statement `v`.
fn written_at(cx: &Context, cells: &[StmtId], index: StmtId) -> Option<StmtId> {
    let constant = |s: StmtId| match cx.stmt(s).ir {
        IR::ConstantInt { value } => Some(value),
        _ => None,
    };
    let tests_index_at = |c: StmtId, k: i64| {
        let test = cx.stmt(c);
        matches!(test.ir, IR::EqI)
            && match test.arguments.as_slice() {
                &[x, y] => (x == index && constant(y) == Some(k)) || (y == index && constant(x) == Some(k)),
                _ => false,
            }
    };
    let mut value = None;
    for (k, &cell) in cells.iter().enumerate() {
        let stmt = cx.stmt(cell);
        let &[c, v, _] = stmt.arguments.as_slice() else { return None };
        if !matches!(stmt.ir, IR::SelectI | IR::SelectF) || !tests_index_at(c, k as i64) || *value.get_or_insert(v) != v {
            return None;
        }
    }
    value
}

struct WrittenReadEmitter {
    /// Each read's lane and the written value it reads.
    written: Vec<(StmtId, StmtId)>,
}

impl Emitter for WrittenReadEmitter {
    fn lane(&mut self, _b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let value = self.written.iter().find(|(stmt, _)| *stmt == lane.stmt).expect("a read of this node").1;
        cx.value(value)
    }
}

/// Every read of the group as a select chain over its index's equality bits
/// (`element::gather_by_select_chain`). Cost `cost::element_by_select_chain`.
pub struct ElementBySelectChain;

/// Every read of the group as the inner product of the cells with its
/// index's equality bits (`element::gather_by_indicators`). Cost
/// `cost::element_by_indicators`.
pub struct ElementByIndicators;

/// Every read of the group through a memory segment holding the cells (the
/// backends' dynamic read: a one-hot gather on halo2-base, a native array
/// read on Noir). Cost `cost::element_by_memory`.
pub struct ElementByMemory;

impl Construction for ElementByOneHot {
    fn name(&self) -> &'static str {
        "gather: element by a recorded one-hot"
    }

    /// One candidate for all the group's reads when each has a one-hot
    /// (fewer nodes for the selection), else one per read that has.
    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::Element {
            return Vec::new();
        }
        let reads: Vec<(&Lane, (GroupId, usize, i64))> = group
            .lanes
            .iter()
            .filter_map(|lane| Some((lane, encoding_source(cx, *lane.own.first()?)?)))
            .collect();
        let candidate = |reads: &[(&Lane, (GroupId, usize, i64))]| {
            let mut pre: Vec<Requirement> =
                reads.iter().map(|(_, (g, _, _))| Requirement::Shared { group: *g, name: ENCODING }).collect();
            pre.sort();
            pre.dedup();
            let cost = reads
                .iter()
                .map(|(_, (g, _, _))| {
                    let onehot_is_free = cx.groups[*g].kind == OpKind::PermutationScatter;
                    cost::element_by_onehot(group.shared.len() as u64, onehot_is_free, cx.target)
                })
                .sum();
            Candidate {
                pre,
                post: reads.iter().map(|(lane, _)| Requirement::Lane(lane.stmt)).collect(),
                cost,
                emitter: Box::new(OneHotReadEmitter {
                    cells: group.shared.clone(),
                    sources: reads.iter().map(|(lane, source)| (lane.stmt, *source)).collect(),
                }),
            }
        };
        if reads.len() == group.lanes.len() {
            return vec![candidate(&reads)];
        }
        reads.iter().map(|read| candidate(std::slice::from_ref(read))).collect()
    }
}

struct OneHotReadEmitter {
    cells: Vec<StmtId>,
    /// Each read's lane and the source of its index's one-hot.
    sources: Vec<(StmtId, (GroupId, usize, i64))>,
}

impl Emitter for OneHotReadEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let cells = cx.values(&self.cells);
        let source = self.sources.iter().find(|(stmt, _)| *stmt == lane.stmt).expect("a read of this node").1;
        let encoding = encoding_at(b, cx, source);
        super::element::gather_by_onehot(b, &cells, &encoding, dtype_of(&cells))
    }
}

/// How [`IndexedReadEmitter`] reads the cells at a lane's index.
#[derive(Clone, Copy)]
enum IndexedRead {
    SelectChain,
    Indicators,
}

impl IndexedRead {
    /// The cost of one read of `n` cells at `index`, read `index_reads`
    /// times in the program.
    fn cost(self, cx: &Context, n: u64, index: StmtId, index_reads: u64) -> u64 {
        match self {
            IndexedRead::SelectChain => {
                cost::element_by_select_chain(n, index_reads, checked_below(cx, index, n), cx.target)
            }
            IndexedRead::Indicators => cost::element_by_indicators(n, index_reads, cx.target),
        }
    }

    /// One candidate for every read of an element group: the sum of the
    /// per-read costs, each read sharing its index's equality tests with
    /// every element read at that index in the program.
    fn candidate(self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::Element {
            return Vec::new();
        }
        let n = group.shared.len() as u64;
        let indices: Vec<StmtId> = group.lanes.iter().map(|l| l.own[0]).collect();
        let cost = indices
            .iter()
            .zip(index_reads(cx, &indices))
            .map(|(&index, reads)| self.cost(cx, n, index, reads))
            .sum();
        vec![Candidate {
            pre: Vec::new(),
            post: lanes(group).collect(),
            cost,
            emitter: Box::new(IndexedReadEmitter { how: self, cells: group.shared.clone() }),
        }]
    }
}

/// Whether the program checks `AssertU128Lt(index, n)`.
fn checked_below(cx: &Context, index: StmtId, n: u64) -> bool {
    cx.users(index).iter().any(|&user| {
        let stmt = cx.stmt(user);
        matches!(stmt.ir, IR::AssertU128Lt)
            && stmt.arguments.first() == Some(&index)
            && matches!(stmt.arguments.get(1).map(|&h| &cx.stmt(h).ir), Some(IR::ConstantInt { value }) if *value as u64 == n)
    })
}

impl Construction for ElementBySelectChain {
    fn name(&self) -> &'static str {
        "gather: element by a select chain"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        IndexedRead::SelectChain.candidate(group, cx)
    }
}

impl Construction for ElementByIndicators {
    fn name(&self) -> &'static str {
        "gather: element by indicators"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        IndexedRead::Indicators.candidate(group, cx)
    }
}

struct IndexedReadEmitter {
    how: IndexedRead,
    cells: Vec<StmtId>,
}

impl Emitter for IndexedReadEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let cells = cx.values(&self.cells);
        let index = cx.value(lane.own[0]);
        let dtype = dtype_of(&cells);
        match self.how {
            IndexedRead::SelectChain => super::element::gather_by_select_chain(b, &cells, &index, dtype),
            IndexedRead::Indicators => super::element::gather_by_indicators(b, &cells, &index, dtype),
        }
    }
}

impl Construction for ElementByMemory {
    fn name(&self) -> &'static str {
        "gather: element through memory"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::Element {
            return Vec::new();
        }
        let reads = group.lanes.len() as u64;
        let mut indices: Vec<StmtId> = group.lanes.iter().filter_map(|l| l.own.first().copied()).collect();
        indices.sort_unstable();
        indices.dedup();
        vec![Candidate {
            pre: Vec::new(),
            post: lanes(group).collect(),
            cost: cost::element_by_memory(group.shared.len() as u64, reads, &index_reads(cx, &indices), cx.target),
            emitter: Box::new(MemoryReadEmitter { cells: group.shared.clone(), segment: None }),
        }]
    }
}

/// For each index, the number of element reads at it over every element
/// group of the program (reads at one index share its check).
fn index_reads(cx: &Context, indices: &[StmtId]) -> Vec<u64> {
    indices
        .iter()
        .map(|index| {
            cx.groups
                .iter()
                .filter(|g| g.kind == OpKind::Element)
                .flat_map(|g| &g.lanes)
                .filter(|lane| lane.own.first() == Some(index))
                .count() as u64
        })
        .collect()
}

struct MemoryReadEmitter {
    cells: Vec<StmtId>,
    segment: Option<(u32, NumberType)>,
}

impl Emitter for MemoryReadEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let cells = cx.values(&self.cells);
        let dtype = dtype_of(&cells);
        self.segment = Some((crate::helpers::segment::alloc_and_write_values(b, &cells, dtype), dtype));
    }

    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let (segment, dtype) = self.segment.expect("started");
        let index = cx.value(lane.own[0]);
        crate::helpers::segment::read_typed(b, segment, &index, dtype)
    }
}

fn dtype_of(cells: &[Value]) -> NumberType {
    if cells.iter().any(|v| matches!(v, Value::Float(_))) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}
