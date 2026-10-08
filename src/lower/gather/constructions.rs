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
use crate::lower::graph::CellSource;
use crate::lower::{
    Bound, Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, GroupId, Lane, OpKind, Requirement,
};
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
pub(crate) type Encodings = Vec<IndexEncoding>;

/// The family's constructions, in the order the selection breaks ties.
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![
        &ScanByRunningProducts,
        &ScanBySelectChain,
        &ScanOfMonotoneFlags,
        &ScanOfMonotoneFlagsByPosition,
        &ScatterByHits,
        &ScatterByWrites,
        &WindowShift,
        &ElementOfWrite,
        &ElementByOneHot,
        &ElementBySelectChain,
        &ElementByIndicators,
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

/// A scan of flags that are monotone along the line: rising (all unset,
/// then all set) or falling (all set, then all unset).
///
/// * Monotone flags come from comparisons `x_k > t` (`>=`, `t < x_k`,
///   `t <= x_k`: rising; `<`, `<=`: falling) of a line `x` with one value
///   `t`, where `NonDecreasing(x)` is established (`Context::established`,
///   an assumption about the input), or from flags stated non-decreasing
///   themselves (`NonDecreasing(flags)` rising, of the reversed flags
///   falling).
/// * Postconditions: the lane.
/// * Rising flags with `s` set: the first set position is `n - s`, NumPy's
///   argmax `n - s` when `s > 0` and 0 otherwise, the last set position
///   `n - 1`. Falling flags: the first set position and the argmax are 0,
///   the last set position `s - 1`. (`FirstSet` and `LastSet` have a set
///   flag where used, the operator's precondition.)
/// * Two candidates: **by a count**, `s` the sum of the flags; and, for
///   flags from comparisons, **by a hinted position**, the count of the
///   flags' complement found in the sorted `x` as `np.searchsorted(x, t)`
///   does (`sort::search::position_in_sorted`: two reads and two
///   comparisons, the flags unread). The second is credited with the
///   flags' comparisons when nothing else reads them (they die).
/// * Soundness: no hint for the count; the position's hint is pinned
///   (`position_in_sorted`). Given the order of `x`, `x_k > t` holds exactly
///   for the `k` at or past the first `x_k > t`, so the flags are rising and
///   their unset prefix has `#{x_k <= t}` positions (the strict case;
///   `#{x_k < t}` for `>=`); falling flags mirror it. The order is the
///   precondition, kept by the selection.
pub struct ScanOfMonotoneFlags;

/// [`ScanOfMonotoneFlags`] by a hinted position in the compared line (its
/// second candidate, a construction of its own so that measurements can
/// force either).
pub struct ScanOfMonotoneFlagsByPosition;

/// How the flags of a scan are monotone, and what they compare.
struct Monotone {
    rising: bool,
    /// For flags comparing a line with one value: the line, the value, and
    /// whether the comparison is strict (`>` or `<`).
    comparison: Option<(Vec<StmtId>, StmtId, bool)>,
    /// The established order the monotonicity rests on.
    order: Fact,
}

/// The comparisons `x_k cmp t` behind the flags, all of one kind with one
/// `t`: `(x, t, rising, strict)`.
fn compared_line(cx: &Context, flags: &[StmtId]) -> Option<(Vec<StmtId>, StmtId, bool, bool)> {
    // `a cmp b` as (a, b, `a > b` or `a >= b`, strict).
    let comparison = |flag: StmtId| {
        let stmt = cx.stmt(flag);
        let &[a, b] = stmt.arguments.as_slice() else { return None };
        match stmt.ir {
            IR::GtI | IR::GtF => Some((a, b, true, true)),
            IR::GteI | IR::GteF => Some((a, b, true, false)),
            IR::LtI | IR::LtF => Some((a, b, false, true)),
            IR::LteI | IR::LteF => Some((a, b, false, false)),
            _ => None,
        }
    };
    let comparisons: Vec<(StmtId, StmtId, bool, bool)> = flags.iter().map(|&f| comparison(f)).collect::<Option<_>>()?;
    // The value `t` is the operand every comparison shares (`x_k cmp t` or
    // `t cmp x_k`, the latter the mirrored order: rising for `t < x_k`).
    let (a0, b0, _, _) = comparisons[0];
    let shared = [a0, b0].into_iter().find(|&t| comparisons.iter().all(|&(a, b, _, _)| (a == t) != (b == t)))?;
    let oriented: Vec<(StmtId, bool, bool)> = comparisons
        .iter()
        .map(|&(a, b, greater, strict)| if b == shared { (a, greater, strict) } else { (b, !greater, strict) })
        .collect();
    let (_, rising, strict) = oriented[0];
    if oriented.iter().any(|&(_, r, s)| (r, s) != (rising, strict)) {
        return None;
    }
    Some((oriented.into_iter().map(|(x, _, _)| x).collect(), shared, rising, strict))
}

fn monotone(group: &Group, cx: &Context) -> Option<Monotone> {
    use crate::lower::{Claim, ListId};
    let flags = &group.shared;
    if flags.len() < 2 {
        return None;
    }
    if let Some((line, t, rising, strict)) = compared_line(cx, flags) {
        let order = Fact::everywhere(Claim::NonDecreasing(ListId::of_stmts(&line)));
        if cx.established(&order) {
            return Some(Monotone { rising, comparison: Some((line, t, strict)), order });
        }
    }
    let rising = Fact::everywhere(Claim::NonDecreasing(ListId::of_stmts(flags)));
    if cx.established(&rising) {
        return Some(Monotone { rising: true, comparison: None, order: rising });
    }
    let reversed: Vec<StmtId> = flags.iter().rev().copied().collect();
    let falling = Fact::everywhere(Claim::NonDecreasing(ListId::of_stmts(&reversed)));
    cx.established(&falling).then_some(Monotone { rising: false, comparison: None, order: falling })
}

impl Construction for ScanOfMonotoneFlags {
    fn name(&self) -> &'static str {
        "gather: scan of monotone flags by a count"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        monotone_candidates(group, cx, false)
    }
}

impl Construction for ScanOfMonotoneFlagsByPosition {
    fn name(&self) -> &'static str {
        "gather: scan of monotone flags by a hinted position"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        monotone_candidates(group, cx, true)
    }
}

fn monotone_candidates(group: &Group, cx: &Context, by_position: bool) -> Vec<Candidate> {
    {
        let Some(kind) = ScanKind::of(group.kind) else { return Vec::new() };
        let Some(monotone) = monotone(group, cx) else { return Vec::new() };
        let n = group.shared.len() as u64;
        let u = cost::units(cx.target);
        let candidate = |how: MonotoneBy, cost: u64| Candidate {
            pre: vec![Requirement::Fact(monotone.order)],
            post: lanes(group).collect(),
            cost,
            emitter: Box::new(MonotoneScanEmitter { kind, rising: monotone.rising, flags: group.shared.clone(), how }),
        };
        if !by_position {
            return vec![candidate(MonotoneBy::Count, n * u.add + u.select)];
        }
        let mut out = Vec::new();
        if let Some((line, t, strict)) = monotone.comparison {
            let position = crate::lower::sort::search::runtime_position_cost(line.len(), 1, cx.target);
            // The flags' comparisons die with the scan when it is their
            // only reader (priced at the full width, or the narrow one of
            // an established range of the line and the value).
            let only_here = group.shared.iter().all(|&f| cx.users(f).iter().all(|u| group.lanes.iter().any(|l| l.stmt == *u)));
            let mut compared = line.clone();
            compared.push(t);
            let width = cx.element_range(&compared).map_or(64, |(lo, hi, _)| 64 - ((hi - lo) as u64).leading_zeros());
            let credit = if only_here { n * crate::lower::checks::comparison_cost(width, cx.target) } else { 0 };
            out.push(candidate(MonotoneBy::Position { line, t, strict }, position.saturating_sub(credit)));
        }
        out
    }
}

/// How [`ScanOfMonotoneFlags`] finds the count of set flags.
#[derive(Clone)]
enum MonotoneBy {
    Count,
    /// The sorted line, the value and the strictness the flags compare.
    Position { line: Vec<StmtId>, t: StmtId, strict: bool },
}

struct MonotoneScanEmitter {
    kind: ScanKind,
    rising: bool,
    flags: Vec<StmtId>,
    how: MonotoneBy,
}

impl Emitter for MonotoneScanEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, _lane: &Lane) -> Value {
        let n = self.flags.len() as i64;
        // The number of set flags.
        let set = match &self.how {
            MonotoneBy::Count => {
                let flags = cx.values(&self.flags);
                let mut total = b.ir_constant_int(0);
                for flag in &flags {
                    let bit = b.ir_int_cast(flag);
                    total = b.ir_add_i(&total, &bit);
                }
                total
            }
            MonotoneBy::Position { line, t, strict } => {
                // Rising flags `x_k > t` are unset for the `#{x_k <= t}`
                // first positions (`>=`: `#{x_k < t}`); falling flags
                // `x_k < t` are set for the `#{x_k < t}` first positions
                // (`<=`: `#{x_k <= t}`).
                let entries = cx.values(line);
                let t = cx.value(*t);
                let right = self.rising == *strict;
                let extended = crate::lower::sort::search::extended_runtime_table(b, &entries);
                let found = crate::lower::sort::search::position_in_sorted(b, &entries, extended, &t, right);
                if self.rising {
                    let len = b.ir_constant_int(n);
                    b.ir_sub_i(&len, &found)
                } else {
                    found
                }
            }
        };
        let len = b.ir_constant_int(n);
        match (self.kind, self.rising) {
            (ScanKind::First, true) => b.ir_sub_i(&len, &set),
            (ScanKind::Argmax, true) => {
                let zero = b.ir_constant_int(0);
                let none = b.ir_equal_i(&set, &zero);
                let first = b.ir_sub_i(&len, &set);
                b.ir_select_i(&none, &zero, &first)
            }
            (ScanKind::Last, true) => b.ir_constant_int(n - 1),
            (ScanKind::First | ScanKind::Argmax, false) => b.ir_constant_int(0),
            (ScanKind::Last, false) => {
                let one = b.ir_constant_int(1);
                b.ir_sub_i(&set, &one)
            }
        }
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
    let wrap = matches!(group.kind, OpKind::Wrap { .. }) && group.lanes.iter().any(|l| l.stmt == root && l.output == 0);
    let encoded = ScanKind::of(group.kind).is_some() || scatter_offset(cx, group).is_some() || wrap;
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
/// per `j` since `p` is a permutation (its precondition, the fact
/// `Permutation(p, 0..n)` the program's `is_permutation` check
/// establishes, `lower::permutation`). The hits are the precondition's own (`emit_is_permutation`;
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
        let permutation = crate::lower::permutation::permutation_fact(&group.shared[..n]);
        // The hits are the permutation check's equality tests when the
        // program checks `p` (deduplicated); an argsort, a permutation by
        // definition, has no check, and the scatter pays for them.
        let checked = cx.sharing_groups(group).any(|g| g.kind == OpKind::IsPermutation && g.shared[..] == group.shared[..n]);
        let tests = if checked { 0 } else { (n * n) as u64 * cost::units(cx.target).eq_const };
        vec![Candidate {
            pre: vec![Requirement::Fact(permutation)],
            post,
            cost: cost::permutation_scatter(n as u64, cx.target) + tests,
            emitter: Box::new(ScatterEmitter { operands: group.shared.clone(), offset, hits: Vec::new(), cells: Vec::new() }),
        }]
    }
}

/// `c[p] = v` for a permutation `p` as `n` writes into a fresh array of `n`
/// cells, its cells read at constant addresses: an array version chain the
/// memory family lowers in the next round (a RAM trace on halo2-base, the
/// native RAM on Noir, or an overlay of selects), linear in `n` where the
/// hits are quadratic.
///
/// * Preconditions: `Permutation(p, 0..n)` (every write lands in the array,
///   and every cell is written exactly once, so its value does not depend
///   on the order of the writes or the initial zeros); on halo2-base the
///   RAM trace (`memory::versions::RAM`, paid once), which the next round's
///   lowering most likely uses (its estimate here).
/// * Postconditions: every cell.
/// * Soundness: the writes and reads are the array semantics the memory
///   constructions implement; with `p` a permutation, cell `j` holds the
///   `v_k` with `p_k = j` (NumPy's `c[p] = v`).
/// * Cost: the memory family's estimate of the chain (`memory::versions`):
///   halo2-base 8 cells per initial row and per access (`n` writes, `n`
///   reads); Noir's native RAM `3n + 3.5n` gates per access.
pub struct ScatterByWrites;

impl Construction for ScatterByWrites {
    fn name(&self) -> &'static str {
        "gather: permutation scatter by writes"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::PermutationScatter {
            return Vec::new();
        }
        let n = group.shared.len() / 2;
        let permutation = crate::lower::permutation::permutation_fact(&group.shared[..n]);
        let (pre, cost) = match cx.target {
            crate::lower::Target::Halo2Base => {
                (vec![Requirement::Fact(permutation), crate::lower::memory::versions::RAM], 8 * 3 * n as u64)
            }
            crate::lower::Target::Noir => {
                let n = n as u64;
                (vec![Requirement::Fact(permutation)], 3 * n + (7 * n).div_ceil(2) * 2 * n)
            }
        };
        vec![Candidate {
            pre,
            post: lanes(group).collect(),
            cost,
            emitter: Box::new(WritesEmitter { operands: group.shared.clone(), version: None }),
        }]
    }
}

struct WritesEmitter {
    operands: Vec<StmtId>,
    /// The array after the writes.
    version: Option<(Value, crate::ir_array::ArrayElem)>,
}

impl Emitter for WritesEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        use crate::ir_array::{ArrayAccess, ArrayElem};
        let values = cx.values(&self.operands);
        let n = values.len() / 2;
        let (p, v) = values.split_at(n);
        let elem = if dtype_of(v) == NumberType::Float { ArrayElem::Float } else { ArrayElem::Int };
        let zeros: Vec<Value> =
            (0..n).map(|_| if elem == ArrayElem::Float { b.ir_constant_float(0.0) } else { b.ir_constant_int(0) }).collect();
        let always = b.ir_constant_bool(true);
        let mut version = b.ir_array_init(&zeros, elem);
        for (pk, vk) in p.iter().zip(v) {
            version = b.ir_array_write(&version, ArrayAccess::flat(n as u32, elem), std::slice::from_ref(pk), vk, &always);
        }
        self.version = Some((version, elem));
    }

    fn lane(&mut self, b: &mut IRBuilder, _cx: &mut EmitCx, lane: &Lane) -> Value {
        let (version, elem) = self.version.clone().expect("begin ran");
        let n = self.operands.len() / 2;
        let j = lane.position.expect("a scatter cell has a position");
        b.ir_array_read(&version, crate::ir_array::ArrayAccess::constant(n as u32, elem, j), &[])
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
    // `[index == k]`: an equality test, or the bit `k` of the NumPy wrap
    // whose position `index` is (`lower::memory::wrap`).
    let wrap_bit_at = |c: StmtId, k: i64| {
        let (IR::ArrayOp(op), Some(group)) = (&cx.stmt(c).ir, cx.group_of_lane(c)) else { return false };
        matches!(op.kind, OpKind::Wrap { .. })
            && op.output == crate::lower::memory::wrap::HIT
            && op.position == u32::try_from(k).ok()
            && group.lanes.iter().any(|l| l.stmt == index && l.output == crate::lower::memory::wrap::POSITION)
    };
    let tests_index_at = |c: StmtId, k: i64| {
        let test = cx.stmt(c);
        let equality = matches!(test.ir, IR::EqI)
            && match test.arguments.as_slice() {
                &[x, y] => (x == index && constant(y) == Some(k)) || (y == index && constant(x) == Some(k)),
                _ => false,
            };
        equality || wrap_bit_at(c, k)
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
                    // A permutation scatter's hits and a wrap's position bits
                    // are cells already; a scan's one-hot is products to build.
                    let onehot_is_free = matches!(cx.groups[*g].kind, OpKind::PermutationScatter | OpKind::Wrap { .. });
                    cost::element_by_onehot(group.line_len() as u64, onehot_is_free, cx.target)
                })
                .sum();
            Candidate {
                pre,
                post: reads.iter().map(|(lane, _)| Requirement::Lane(lane.stmt)).collect(),
                cost,
                emitter: Box::new(OneHotReadEmitter {
                    cells: group.cells(),
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
    cells: CellSource,
    /// Each read's lane and the source of its index's one-hot.
    sources: Vec<(StmtId, (GroupId, usize, i64))>,
}

impl Emitter for OneHotReadEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let cells = cx.cells(b, &self.cells);
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

    /// The candidates for every read of an element group: the sum of the
    /// per-read costs, each read sharing its index's equality tests with
    /// every element read at that index in the program. For indicators, a
    /// second candidate also asserts that the indicators of each index sum
    /// to 1 (`element::assert_one_hot`), which proves `0 <= index < n`: it
    /// is offered only when the program checks exactly that unconditionally
    /// (`AssertU128Lt(index, n)`), so it rejects nothing the program
    /// accepts, and the selection may then drop the program's check.
    fn candidate(self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::Element {
            return Vec::new();
        }
        let n = group.line_len() as u64;
        let indices: Vec<StmtId> = group.lanes.iter().map(|l| l.own[0]).collect();
        let cost: u64 = indices
            .iter()
            .zip(index_reads(cx, group, &indices))
            .map(|(&index, reads)| self.cost(cx, n, index, reads))
            .sum();
        let candidate = |proves: bool| {
            let mut distinct = indices.clone();
            distinct.sort_unstable();
            distinct.dedup();
            let facts: Vec<Requirement> = if proves {
                distinct
                    .iter()
                    .map(|&x| Requirement::Fact(Fact::everywhere(Claim::InRange { x, lo: 0, hi: Bound::Const(n as i64) })))
                    .collect()
            } else {
                Vec::new()
            };
            let checks = if proves { distinct.len() as u64 * cost::one_hot_check(n, cx.target) } else { 0 };
            Candidate {
                pre: Vec::new(),
                post: lanes(group).chain(facts).collect(),
                cost: cost + checks,
                emitter: Box::new(IndexedReadEmitter { how: self, cells: group.cells(), proves, checked: Vec::new() }),
            }
        };
        let mut out = vec![candidate(false)];
        if matches!(self, IndexedRead::Indicators) && indices.iter().all(|&index| checked_below(cx, index, n)) {
            out.push(candidate(true));
        }
        out
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
    cells: CellSource,
    /// Assert each index's one-hot (the candidate proves its range).
    proves: bool,
    /// The indices whose one-hot is asserted already.
    checked: Vec<StmtId>,
}

impl Emitter for IndexedReadEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let cells = cx.cells(b, &self.cells);
        let index = cx.value(lane.own[0]);
        let dtype = dtype_of(&cells);
        match self.how {
            IndexedRead::SelectChain => super::element::gather_by_select_chain(b, &cells, &index, dtype),
            IndexedRead::Indicators => {
                if self.proves && !self.checked.contains(&lane.own[0]) {
                    self.checked.push(lane.own[0]);
                    super::element::assert_one_hot(b, &index, cells.len());
                }
                super::element::gather_by_indicators(b, &cells, &index, dtype)
            }
        }
    }
}

/// For each index, the number of element reads at it over the element
/// groups that share costs with `group` (reads at one index share its
/// check; `Context::sharing_groups`).
fn index_reads(cx: &Context, group: &Group, indices: &[StmtId]) -> Vec<u64> {
    indices
        .iter()
        .map(|index| {
            cx.sharing_groups(group)
                .filter(|g| g.kind == OpKind::Element)
                .flat_map(|g| &g.lanes)
                .filter(|lane| lane.own.first() == Some(index))
                .count() as u64
        })
        .collect()
}

fn dtype_of(cells: &[Value]) -> NumberType {
    if cells.iter().any(|v| matches!(v, Value::Float(_))) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}
