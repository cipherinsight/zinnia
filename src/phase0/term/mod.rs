//! The hash-consed term language of `revision/phase0/definition.md` §3.1.
//!
//! Phase 0 records lengths, counts and definedness conditions as terms of
//! integer linear arithmetic. This module is the only place those terms are
//! built, evaluated and printed.
//!
//! # One arena, two id newtypes
//!
//! Integer terms and Boolean terms live in one [`TermArena`]. Nodes of both
//! sorts share a single `Vec<Node>` and a single hash-cons table, but the
//! indices handed out are two distinct newtypes, [`TermId`] for integer sort
//! and [`BoolId`] for Boolean sort. The alternative — one `TermId` plus a
//! run-time sort check — was rejected: the transfer rules of `rules/` pass
//! dozens of terms between helpers (`norm`, `bjoin`, `range_len`), and with
//! two newtypes the Rust type checker rejects a sort confusion at the call
//! site instead of a rule returning `Reject` at run time. The cost is that a
//! few constructors exist twice (`ite` for integer results, `bool_ite` for
//! Boolean results); that is a handful of lines.
//!
//! Terms of integer sort are unbounded mathematical integers. Sort
//! information (`Idx` versus `I64`) lives on the *variables* ([`VarDecl`])
//! and on descriptors, never on a term node.
//!
//! # Invariants
//!
//! - Hash-consing: structurally equal nodes are one node, so `TermId`
//!   equality is structural equality. Every simplification below relies on
//!   this.
//! - A node's children always have strictly smaller arena indices than the
//!   node itself, because `intern` appends. Evaluation and printing exploit
//!   this to work bottom-up without recursion.
//! - The constructors apply exactly the simplifications §3.1 allows:
//!   constant folding, `ite(true, a, b) -> a` / `ite(false, a, b) -> b`, and
//!   absorption of constants under `min` / `max`. Identity folds on
//!   hash-consed-equal children (`t = t -> true`, `ite(b, t, t) -> t`,
//!   `t + 0 -> t`, ...) are constant folding on a syntactically known value
//!   and are applied too. Nothing else: further simplification changes memo
//!   hit rates but not phase 1 answers (§3.1).

use std::collections::HashMap;

/// The sort of a variable of $\Phi$ (`definition.md` §3.2, §3.4).
///
/// `Idx` is an unbounded Python integer used for lengths, counts and loop
/// counters; `I64` is a machine integer read out of an array, whose
/// arithmetic wraps (§3.2); `Bool` is a Boolean variable. Float, Complex and
/// String scalars carry no term at all, so they have no sort here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sort {
    Idx,
    I64,
    Bool,
}

impl Sort {
    pub fn is_bool(self) -> bool {
        matches!(self, Sort::Bool)
    }

    /// The SMT-LIB 2 sort name used by [`TermArena::to_smtlib`].
    pub fn smt(self) -> &'static str {
        if self.is_bool() {
            "Bool"
        } else {
            "Int"
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Sort::Idx => "Idx",
            Sort::I64 => "I64",
            Sort::Bool => "Bool",
        }
    }
}

/// Index of a variable of $\Phi$. One numbering covers integer and Boolean
/// variables; the sort is in the variable's [`VarDecl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub u32);

/// Index of an integer-sorted node in a [`TermArena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TermId(u32);

/// Index of a Boolean-sorted node in a [`TermArena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BoolId(u32);

impl TermId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl BoolId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A declared variable: the name used by every printer and the sort.
#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: String,
    pub sort: Sort,
}

/// A node of the term DAG. Integer-sorted variants come first; the
/// discriminant is never exposed, only the two id newtypes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Node {
    // ── integer sort ────────────────────────────────────────────────
    Const(i128),
    Var(VarId),
    Add(TermId, TermId),
    Sub(TermId, TermId),
    /// `k * t`, `k` a constant (§3.1 forbids a product of two non-constants).
    Mul(i128, TermId),
    /// `t div k`, `k` a non-zero constant, floor semantics (Python `//`).
    Div(TermId, i128),
    /// `t mod k`, `k` a non-zero constant, floor semantics (Python `%`).
    Mod(TermId, i128),
    Min(TermId, TermId),
    Max(TermId, TermId),
    Ite(BoolId, TermId, TermId),
    // ── Boolean sort ────────────────────────────────────────────────
    True,
    False,
    BVar(VarId),
    Eq(TermId, TermId),
    Lt(TermId, TermId),
    Le(TermId, TermId),
    Not(BoolId),
    And(BoolId, BoolId),
    Or(BoolId, BoolId),
}

/// The structure of one integer-sorted term, as [`TermArena::int_node`]
/// hands it out.
///
/// It mirrors the integer half of the private node enum. An `ite` appears
/// without its condition, because the one consumer — interval evaluation —
/// takes the union of the two arms and never looks at the condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntNode {
    Const(i128),
    Var(VarId),
    Add(TermId, TermId),
    Sub(TermId, TermId),
    /// `k * t`, `k` a constant.
    Mul(i128, TermId),
    /// `t div k`, floor semantics.
    Div(TermId, i128),
    /// `t mod k`, floor semantics.
    Mod(TermId, i128),
    Min(TermId, TermId),
    Max(TermId, TermId),
    /// `ite(_, a, b)`, condition omitted.
    Ite(TermId, TermId),
    /// Not an integer-sorted node; unreachable through a [`TermId`].
    Opaque,
}

/// `2^63`, the offset of [`TermArena::wrap64`] (`definition.md` §3.2).
const TWO_POW_63: i128 = 1 << 63;
/// `2^64`, the modulus of [`TermArena::wrap64`].
const TWO_POW_64: i128 = 1 << 64;
/// Inclusive range of a signed 64-bit machine integer (`Sort::I64`).
pub const I64_MIN: i128 = -TWO_POW_63;
pub const I64_MAX: i128 = TWO_POW_63 - 1;

/// Python's `//` on `i128`. Rust's `/` truncates toward zero; Python floors.
pub fn floor_div(a: i128, b: i128) -> i128 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Python's `%` on `i128`: the result takes the sign of the divisor.
pub fn floor_mod(a: i128, b: i128) -> i128 {
    a - floor_div(a, b) * b
}

/// The term DAG together with its variable table.
///
/// The variable table lives here rather than in [`crate::phase0::phi::Phi`]
/// because every printer (pretty, SMT-LIB, JSON) needs names and sorts.
/// `Phi` holds the range and provenance of the same `VarId`s and keeps its
/// own vector in step with this one.
pub struct TermArena {
    nodes: Vec<Node>,
    memo: HashMap<Node, u32>,
    vars: Vec<VarDecl>,
    t_true: BoolId,
    t_false: BoolId,
}

impl Default for TermArena {
    fn default() -> Self {
        Self::new()
    }
}

impl TermArena {
    pub fn new() -> Self {
        let mut arena = TermArena {
            nodes: Vec::new(),
            memo: HashMap::new(),
            vars: Vec::new(),
            // Patched immediately below; `BoolId(0)` is a placeholder that is
            // overwritten before any caller can observe it.
            t_true: BoolId(0),
            t_false: BoolId(0),
        };
        arena.t_true = BoolId(arena.intern(Node::True));
        arena.t_false = BoolId(arena.intern(Node::False));
        arena
    }

    fn intern(&mut self, node: Node) -> u32 {
        if let Some(&id) = self.memo.get(&node) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.memo.insert(node.clone(), id);
        self.nodes.push(node);
        id
    }

    /// How many distinct nodes the arena holds, which is the size the
    /// export reports and the harnesses budget against
    /// (`consistency`'s `MAX_ARENA_NODES`). Named like
    /// [`TermArena::var_count`], not `len`: an arena is not a collection of
    /// the caller's values and never "empty".
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ── variables ───────────────────────────────────────────────────

    /// Declare a variable. Only `Phi::fresh_var` should call this, so that
    /// every variable also gets its range constraint at introduction (§3.4).
    pub fn declare_var(&mut self, name: String, sort: Sort) -> VarId {
        let id = VarId(self.vars.len() as u32);
        self.vars.push(VarDecl { name, sort });
        id
    }

    pub fn var_count(&self) -> usize {
        self.vars.len()
    }

    pub fn var_decl(&self, v: VarId) -> &VarDecl {
        &self.vars[v.0 as usize]
    }

    pub fn var_name(&self, v: VarId) -> &str {
        &self.vars[v.0 as usize].name
    }

    /// The variable a term *is*, when the term is a bare variable. Used to
    /// read back the range `Phi` recorded for it (§3.4).
    pub fn var_of(&self, t: TermId) -> Option<VarId> {
        match self.nodes[t.index()] {
            Node::Var(v) => Some(v),
            _ => None,
        }
    }

    /// The structure of an integer-sorted term, for the interval evaluator
    /// of [`crate::phase0::interval`].
    ///
    /// Terms are otherwise opaque outside this module: the rules build them
    /// and the printers read them. Interval evaluation is the one consumer
    /// that has to walk the DAG, so it gets this read-only view rather than
    /// a second copy of the node enum.
    pub fn int_node(&self, t: TermId) -> IntNode {
        match self.nodes[t.index()] {
            Node::Const(n) => IntNode::Const(n),
            Node::Var(v) => IntNode::Var(v),
            Node::Add(a, b) => IntNode::Add(a, b),
            Node::Sub(a, b) => IntNode::Sub(a, b),
            Node::Mul(k, t) => IntNode::Mul(k, t),
            Node::Div(t, k) => IntNode::Div(t, k),
            Node::Mod(t, k) => IntNode::Mod(t, k),
            Node::Min(a, b) => IntNode::Min(a, b),
            Node::Max(a, b) => IntNode::Max(a, b),
            Node::Ite(_, a, b) => IntNode::Ite(a, b),
            // Unreachable: a `TermId` names an integer-sorted node, which is
            // what the two id newtypes keep apart. `Opaque` rather than a
            // panic, and the evaluator reads it as "no information".
            _ => IntNode::Opaque,
        }
    }

    // ── integer constructors ────────────────────────────────────────

    pub fn constant(&mut self, n: i128) -> TermId {
        TermId(self.intern(Node::Const(n)))
    }

    /// The integer-sorted occurrence of a variable.
    pub fn var(&mut self, v: VarId) -> TermId {
        TermId(self.intern(Node::Var(v)))
    }

    fn as_const(&self, t: TermId) -> Option<i128> {
        match self.nodes[t.index()] {
            Node::Const(n) => Some(n),
            _ => None,
        }
    }

    pub fn add(&mut self, a: TermId, b: TermId) -> TermId {
        match (self.as_const(a), self.as_const(b)) {
            (Some(x), Some(y)) => {
                if let Some(s) = x.checked_add(y) {
                    return self.constant(s);
                }
            }
            (Some(0), _) => return b,
            (_, Some(0)) => return a,
            _ => {}
        }
        TermId(self.intern(Node::Add(a, b)))
    }

    pub fn sub(&mut self, a: TermId, b: TermId) -> TermId {
        if a == b {
            return self.constant(0);
        }
        match (self.as_const(a), self.as_const(b)) {
            (Some(x), Some(y)) => {
                if let Some(s) = x.checked_sub(y) {
                    return self.constant(s);
                }
            }
            (_, Some(0)) => return a,
            _ => {}
        }
        TermId(self.intern(Node::Sub(a, b)))
    }

    /// `k * t`. §3.1 allows no product of two non-constant terms.
    pub fn mul_const(&mut self, k: i128, t: TermId) -> TermId {
        if k == 0 {
            return self.constant(0);
        }
        if k == 1 {
            return t;
        }
        if let Some(n) = self.as_const(t) {
            if let Some(p) = k.checked_mul(n) {
                return self.constant(p);
            }
        }
        // Fold nested constant factors: k * (k2 * u) = (k * k2) * u.
        if let Node::Mul(k2, u) = self.nodes[t.index()] {
            if let Some(k3) = k.checked_mul(k2) {
                return self.mul_const(k3, u);
            }
        }
        TermId(self.intern(Node::Mul(k, t)))
    }

    /// `t div k` with Python floor semantics. `None` when `k == 0`: the
    /// caller turns that into a rejection (`E-Arith-Shape`'s premise).
    pub fn div_const(&mut self, t: TermId, k: i128) -> Option<TermId> {
        if k == 0 {
            return None;
        }
        if k == 1 {
            return Some(t);
        }
        if let Some(n) = self.as_const(t) {
            return Some(self.constant(floor_div(n, k)));
        }
        Some(TermId(self.intern(Node::Div(t, k))))
    }

    /// `t mod k` with Python floor semantics. `None` when `k == 0`.
    pub fn mod_const(&mut self, t: TermId, k: i128) -> Option<TermId> {
        if k == 0 {
            return None;
        }
        if k == 1 || k == -1 {
            return Some(self.constant(0));
        }
        if let Some(n) = self.as_const(t) {
            return Some(self.constant(floor_mod(n, k)));
        }
        Some(TermId(self.intern(Node::Mod(t, k))))
    }

    pub fn min(&mut self, a: TermId, b: TermId) -> TermId {
        if a == b {
            return a;
        }
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            return self.constant(x.min(y));
        }
        if let Some(t) = self.absorb(a, b, /* under_min = */ true) {
            return t;
        }
        TermId(self.intern(Node::Min(a, b)))
    }

    pub fn max(&mut self, a: TermId, b: TermId) -> TermId {
        if a == b {
            return a;
        }
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            return self.constant(x.max(y));
        }
        if let Some(t) = self.absorb(a, b, /* under_min = */ false) {
            return t;
        }
        TermId(self.intern(Node::Max(a, b)))
    }

    /// The two folds §3.1 calls "absorption of constants under min and max",
    /// applied to `min(a, b)` (`under_min`) or `max(a, b)`:
    ///
    /// - lattice absorption, `max(a, min(a, y)) = a` and its mirror, which is
    ///   what collapses `norm(0, d) = max(0, min(d, 0))` to `0` (§8.1);
    /// - merging a constant into a nested one, `max(k1, max(t, k2)) =
    ///   max(t, max(k1, k2))`, which collapses the doubled `max(0, ·)` that
    ///   `E-Slice` builds.
    ///
    /// Returns `None` when neither fold applies.
    fn absorb(&mut self, a: TermId, b: TermId, under_min: bool) -> Option<TermId> {
        for (x, y) in [(a, b), (b, a)] {
            // Absorption: the dual operator applied to `x` itself.
            match self.nodes[y.index()] {
                Node::Max(p, q) if under_min && (p == x || q == x) => return Some(x),
                Node::Min(p, q) if !under_min && (p == x || q == x) => return Some(x),
                _ => {}
            }
            // Constant merge into a nested same-operator node.
            let Some(kx) = self.as_const(x) else { continue };
            let (p, q) = match self.nodes[y.index()] {
                Node::Min(p, q) if under_min => (p, q),
                Node::Max(p, q) if !under_min => (p, q),
                _ => continue,
            };
            for first in [true, false] {
                let (inner, other) = if first { (p, q) } else { (q, p) };
                let Some(ki) = self.as_const(inner) else {
                    continue;
                };
                let merged = if under_min { kx.min(ki) } else { kx.max(ki) };
                // The nested constant already dominates: `y` is the answer,
                // unchanged, which keeps the term the caller built.
                if merged == ki {
                    return Some(y);
                }
                let k = self.constant(merged);
                // Rebuild in the nested node's argument order.
                let (np, nq) = if first { (k, other) } else { (other, k) };
                return Some(if under_min {
                    self.min(np, nq)
                } else {
                    self.max(np, nq)
                });
            }
        }
        None
    }

    pub fn ite(&mut self, c: BoolId, a: TermId, b: TermId) -> TermId {
        if c == self.t_true || a == b {
            return a;
        }
        if c == self.t_false {
            return b;
        }
        TermId(self.intern(Node::Ite(c, a, b)))
    }

    /// $\mathrm{wrap}_{64}(t)=((t+2^{63})\bmod 2^{64})-2^{63}$ (§3.2).
    pub fn wrap64(&mut self, t: TermId) -> TermId {
        let off = self.constant(TWO_POW_63);
        let shifted = self.add(t, off);
        // `2^64` is a non-zero constant, so `mod_const` cannot return `None`.
        let wrapped = self
            .mod_const(shifted, TWO_POW_64)
            .expect("2^64 is a non-zero modulus");
        self.sub(wrapped, off)
    }

    // ── Boolean constructors ────────────────────────────────────────

    pub fn tt(&self) -> BoolId {
        self.t_true
    }

    pub fn ff(&self) -> BoolId {
        self.t_false
    }

    pub fn bool_const(&self, b: bool) -> BoolId {
        if b {
            self.t_true
        } else {
            self.t_false
        }
    }

    /// The Boolean-sorted occurrence of a variable.
    pub fn bool_var(&mut self, v: VarId) -> BoolId {
        BoolId(self.intern(Node::BVar(v)))
    }

    pub fn eq(&mut self, a: TermId, b: TermId) -> BoolId {
        if a == b {
            return self.t_true;
        }
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            return self.bool_const(x == y);
        }
        BoolId(self.intern(Node::Eq(a, b)))
    }

    pub fn lt(&mut self, a: TermId, b: TermId) -> BoolId {
        if a == b {
            return self.t_false;
        }
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            return self.bool_const(x < y);
        }
        BoolId(self.intern(Node::Lt(a, b)))
    }

    pub fn le(&mut self, a: TermId, b: TermId) -> BoolId {
        if a == b {
            return self.t_true;
        }
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            return self.bool_const(x <= y);
        }
        BoolId(self.intern(Node::Le(a, b)))
    }

    pub fn not(&mut self, b: BoolId) -> BoolId {
        if b == self.t_true {
            return self.t_false;
        }
        if b == self.t_false {
            return self.t_true;
        }
        if let Node::Not(inner) = self.nodes[b.index()] {
            return inner;
        }
        BoolId(self.intern(Node::Not(b)))
    }

    pub fn and(&mut self, a: BoolId, b: BoolId) -> BoolId {
        if a == self.t_false || b == self.t_false {
            return self.t_false;
        }
        if a == self.t_true || a == b {
            return b;
        }
        if b == self.t_true {
            return a;
        }
        BoolId(self.intern(Node::And(a, b)))
    }

    pub fn or(&mut self, a: BoolId, b: BoolId) -> BoolId {
        if a == self.t_true || b == self.t_true {
            return self.t_true;
        }
        if a == self.t_false || a == b {
            return b;
        }
        if b == self.t_false {
            return a;
        }
        BoolId(self.intern(Node::Or(a, b)))
    }

    /// `a ⇒ b`, written as `¬a ∨ b`; §3.1 has no implication connective.
    pub fn implies(&mut self, a: BoolId, b: BoolId) -> BoolId {
        let na = self.not(a);
        self.or(na, b)
    }

    /// A Boolean-result conditional. §3.1's grammar has `ite` only at integer
    /// sort, so a Boolean merge (`definition.md` §6.1, Scalar merge with
    /// `τ = Bool`) is written `(c ∧ a) ∨ (¬c ∧ b)`.
    pub fn bool_ite(&mut self, c: BoolId, a: BoolId, b: BoolId) -> BoolId {
        let hi = self.and(c, a);
        let nc = self.not(c);
        let lo = self.and(nc, b);
        self.or(hi, lo)
    }

    /// Conjunction of a list; `true` when empty.
    pub fn all(&mut self, items: &[BoolId]) -> BoolId {
        let mut acc = self.t_true;
        for &b in items {
            acc = self.and(acc, b);
        }
        acc
    }

    /// Disjunction of a list; `false` when empty.
    pub fn any(&mut self, items: &[BoolId]) -> BoolId {
        let mut acc = self.t_false;
        for &b in items {
            acc = self.or(acc, b);
        }
        acc
    }

    // ── queries used by the rules ───────────────────────────────────

    /// The constant value of an integer term, when it is a literal.
    pub fn constant_value(&self, t: TermId) -> Option<i128> {
        self.as_const(t)
    }

    pub fn is_true(&self, b: BoolId) -> bool {
        b == self.t_true
    }

    pub fn is_false(&self, b: BoolId) -> bool {
        b == self.t_false
    }
}

// ---------------------------------------------------------------------------
// Evaluation under an assignment (`definition.md` §7, adequacy)
// ---------------------------------------------------------------------------

/// $\theta(u)$: an actual value for every variable of $\Phi$. Used by the
/// unit tests and mirrored by `testing/phase0/terms.py` for the adequacy
/// harness (`acceptance.md` §2.3).
#[derive(Debug, Clone, Default)]
pub struct Assignment {
    ints: HashMap<VarId, i128>,
    bools: HashMap<VarId, bool>,
}

impl Assignment {
    pub fn set_int(&mut self, v: VarId, n: i128) -> &mut Self {
        self.ints.insert(v, n);
        self
    }

    pub fn set_bool(&mut self, v: VarId, b: bool) -> &mut Self {
        self.bools.insert(v, b);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    /// No value for an integer variable.
    UnboundInt(VarId),
    /// No value for a Boolean variable.
    UnboundBool(VarId),
    /// The exact `i128` computation overflowed.
    Overflow,
}

#[derive(Debug, Clone, Copy)]
enum Val {
    I(i128),
    B(bool),
}

impl TermArena {
    fn children(node: &Node, out: &mut Vec<u32>) {
        out.clear();
        match *node {
            Node::Const(_) | Node::Var(_) | Node::True | Node::False | Node::BVar(_) => {}
            Node::Add(a, b) | Node::Sub(a, b) | Node::Min(a, b) | Node::Max(a, b) => {
                out.extend([a.0, b.0])
            }
            Node::Eq(a, b) | Node::Lt(a, b) | Node::Le(a, b) => out.extend([a.0, b.0]),
            Node::Mul(_, t) | Node::Div(t, _) | Node::Mod(t, _) => out.push(t.0),
            Node::Not(b) => out.push(b.0),
            Node::And(a, b) | Node::Or(a, b) => out.extend([a.0, b.0]),
            Node::Ite(c, a, b) => out.extend([c.0, a.0, b.0]),
        }
    }

    /// Nodes the node `root` depends on, itself included. Children always
    /// have smaller indices, so one downward sweep suffices.
    fn reachable(&self, root: u32) -> Vec<bool> {
        let mut need = vec![false; root as usize + 1];
        need[root as usize] = true;
        let mut kids = Vec::with_capacity(3);
        for i in (0..=root as usize).rev() {
            if !need[i] {
                continue;
            }
            Self::children(&self.nodes[i], &mut kids);
            for &c in &kids {
                need[c as usize] = true;
            }
        }
        need
    }

    fn eval_all(&self, root: u32, a: &Assignment) -> Result<Vec<Option<Val>>, EvalError> {
        let need = self.reachable(root);
        let mut vals: Vec<Option<Val>> = vec![None; need.len()];
        for i in 0..need.len() {
            if !need[i] {
                continue;
            }
            // Children were filled on an earlier iteration (their arena
            // index is smaller), so the lookups below always hit.
            let value = {
                let get_i = |id: TermId| match vals[id.index()] {
                    Some(Val::I(n)) => n,
                    _ => unreachable!("an integer-sorted child is evaluated first"),
                };
                let get_b = |id: BoolId| match vals[id.index()] {
                    Some(Val::B(b)) => b,
                    _ => unreachable!("a Boolean-sorted child is evaluated first"),
                };
                self.eval_node(&self.nodes[i], a, &get_i, &get_b)?
            };
            vals[i] = Some(value);
        }
        Ok(vals)
    }

    fn eval_node(
        &self,
        node: &Node,
        a: &Assignment,
        int_of: &dyn Fn(TermId) -> i128,
        bool_of: &dyn Fn(BoolId) -> bool,
    ) -> Result<Val, EvalError> {
        let ck = |r: Option<i128>| r.ok_or(EvalError::Overflow).map(Val::I);
        Ok(match *node {
            Node::Const(n) => Val::I(n),
            Node::Var(v) => Val::I(*a.ints.get(&v).ok_or(EvalError::UnboundInt(v))?),
            Node::Add(x, y) => ck(int_of(x).checked_add(int_of(y)))?,
            Node::Sub(x, y) => ck(int_of(x).checked_sub(int_of(y)))?,
            Node::Mul(k, t) => ck(k.checked_mul(int_of(t)))?,
            Node::Div(t, k) => Val::I(floor_div(int_of(t), k)),
            Node::Mod(t, k) => Val::I(floor_mod(int_of(t), k)),
            Node::Min(x, y) => Val::I(int_of(x).min(int_of(y))),
            Node::Max(x, y) => Val::I(int_of(x).max(int_of(y))),
            Node::Ite(c, x, y) => Val::I(if bool_of(c) { int_of(x) } else { int_of(y) }),
            Node::True => Val::B(true),
            Node::False => Val::B(false),
            Node::BVar(v) => Val::B(*a.bools.get(&v).ok_or(EvalError::UnboundBool(v))?),
            Node::Eq(x, y) => Val::B(int_of(x) == int_of(y)),
            Node::Lt(x, y) => Val::B(int_of(x) < int_of(y)),
            Node::Le(x, y) => Val::B(int_of(x) <= int_of(y)),
            Node::Not(b) => Val::B(!bool_of(b)),
            Node::And(x, y) => Val::B(bool_of(x) && bool_of(y)),
            Node::Or(x, y) => Val::B(bool_of(x) || bool_of(y)),
        })
    }

    /// Value of an integer term under `a`. Cost is linear in the arena
    /// prefix below the term, not in the term's tree size.
    pub fn eval(&self, t: TermId, a: &Assignment) -> Result<i128, EvalError> {
        match self.eval_all(t.0, a)?[t.index()] {
            Some(Val::I(n)) => Ok(n),
            _ => unreachable!("a TermId names an integer-sorted node"),
        }
    }

    /// Value of a Boolean term under `a`.
    pub fn eval_bool(&self, b: BoolId, a: &Assignment) -> Result<bool, EvalError> {
        match self.eval_all(b.0, a)?[b.index()] {
            Some(Val::B(v)) => Ok(v),
            _ => unreachable!("a BoolId names a Boolean-sorted node"),
        }
    }
}

mod print;
