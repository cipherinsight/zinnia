//! Descriptors and value keys: `definition.md` §3.2, §3.3 and
//! `acceptance.md` §2.2.
//!
//! A descriptor is the non-relational half of the abstract state: one per SSA
//! value, recording rank, dtype, the dimension terms, the definedness
//! condition and the execution condition. It carries no capacity, no
//! representation and no number computed by phase 0 (§3.3).
//!
//! The second half of this file is the value-key scheme. Phase 0 and the
//! Python reference evaluator (`testing/phase0/refeval.py`) must agree on a
//! name for every SSA value without using source positions, because the AST
//! JSON never carries them (`acceptance.md` §2.1). [`ValueKey`] is that name
//! and [`KeyCursor`] is what produces it during a traversal.

use crate::phase0::term::{BoolId, TermId, I64_MAX, I64_MIN};
use crate::types::{NumberType, ZinniaType};

// ---------------------------------------------------------------------------
// Dtypes
// ---------------------------------------------------------------------------

/// Element dtype of a descriptor (`definition.md` §3.2: "Idx, I64, Bool,
/// Field, ...").
///
/// It mirrors the baseline's [`NumberType`] (`Integer`, `Float`, `Complex`)
/// and splits `Integer` into the two sorts phase 0 must tell apart: `Idx`,
/// an unbounded Python integer used for lengths, counts and loop counters,
/// and `I64`, a machine integer read out of an array, whose arithmetic wraps
/// (§3.2). `Bool`, `String` and `None` complete the baseline's `Value` kinds
/// that can appear as a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Idx,
    I64,
    Bool,
    Float,
    Complex,
    String,
    None,
}

/// Dtype families. A merge requires equal families (`definition.md` §6.1,
/// "rank or dtype-family mismatch at a merge" in §1's rejection list).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeFamily {
    Integer,
    Float,
    Complex,
    String,
    None,
}

impl Dtype {
    pub fn family(self) -> DtypeFamily {
        match self {
            // The front end aliases `Boolean` to `Integer` (`coverage.md`
            // §4), so `Bool` and the integer sorts are one family.
            Dtype::Idx | Dtype::I64 | Dtype::Bool => DtypeFamily::Integer,
            Dtype::Float => DtypeFamily::Float,
            Dtype::Complex => DtypeFamily::Complex,
            Dtype::String => DtypeFamily::String,
            Dtype::None => DtypeFamily::None,
        }
    }

    /// Whether a scalar of this dtype carries a term at all.
    ///
    /// `Complex` and `String` carry none (`coverage.md` §4). `Float` carries
    /// none either: $\Phi$ is integer linear arithmetic (§3.1) and a
    /// real-sorted variable is not in the term language. See the Deviations
    /// section of `implementation.md`.
    pub fn carries_term(self) -> bool {
        matches!(self, Dtype::Idx | Dtype::I64 | Dtype::Bool)
    }

    /// The dtype range of §3.4, used for a fresh data variable. `Idx` is an
    /// unbounded Python integer, so it has none; intermediate values have no
    /// default upper bound (§3.4).
    pub fn value_range(self) -> (Option<i128>, Option<i128>) {
        match self {
            Dtype::I64 => (Some(I64_MIN), Some(I64_MAX)),
            _ => (None, None),
        }
    }

    /// Whether arithmetic on this dtype wraps, i.e. needs
    /// [`crate::phase0::term::TermArena::wrap64`] (§3.2).
    pub fn wraps(self) -> bool {
        matches!(self, Dtype::I64)
    }

    /// NumPy-style promotion for `E-Map` and `E-Zip`. `None` when the two
    /// dtypes have nothing to promote to, which the caller rejects.
    pub fn promote(a: Dtype, b: Dtype) -> Option<Dtype> {
        if a == b {
            return Some(a);
        }
        let order = |d: Dtype| match d {
            Dtype::Bool => Some(0u8),
            Dtype::Idx => Some(1),
            Dtype::I64 => Some(2),
            Dtype::Float => Some(3),
            Dtype::Complex => Some(4),
            Dtype::String | Dtype::None => None,
        };
        let (oa, ob) = (order(a)?, order(b)?);
        Some(if oa >= ob { a } else { b })
    }

    /// The dtype object this element dtype denotes: `np.float64` for
    /// `Float`, `np.bool_` for `Bool`, and so on.
    ///
    /// This and [`Dtype::from_zinnia_type`] are the one two-way table
    /// between phase 0's element dtypes and the front end's [`ZinniaType`].
    /// The two integer sorts of §3.2 collapse to one `Integer` here, because
    /// the front end has only one; going back gives `I64`, the sort an array
    /// element has.
    pub fn to_zinnia_type(self) -> ZinniaType {
        match self {
            Dtype::Float => ZinniaType::Float,
            Dtype::Complex => ZinniaType::Complex,
            Dtype::Bool => ZinniaType::Boolean,
            _ => ZinniaType::Integer,
        }
    }

    /// The element dtype a [`ZinniaType`] names, or `None` when it names no
    /// element dtype at all (a container, a class, `None`).
    pub fn from_zinnia_type(t: &ZinniaType) -> Option<Dtype> {
        match t {
            ZinniaType::Float => Some(Dtype::Float),
            ZinniaType::Complex => Some(Dtype::Complex),
            ZinniaType::Boolean => Some(Dtype::Bool),
            ZinniaType::Integer => Some(Dtype::I64),
            _ => None,
        }
    }

    /// The element dtype of an array whose annotation says [`NumberType`].
    /// Integer arrays hold machine integers, hence `I64`.
    pub fn from_number_type(n: NumberType) -> Dtype {
        match n {
            NumberType::Integer => Dtype::I64,
            NumberType::Float => Dtype::Float,
            NumberType::Complex => Dtype::Complex,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::Idx => "Idx",
            Dtype::I64 => "I64",
            Dtype::Bool => "Bool",
            Dtype::Float => "Float",
            Dtype::Complex => "Complex",
            Dtype::String => "String",
            Dtype::None => "None",
        }
    }
}

/// Where a scalar's value comes from (`definition.md` §3.2).
///
/// `Shape` means the term is built from the parameters in $S$, shape terms
/// and other shape scalars; `Data` means it depends on element contents, and
/// then the term is a fresh variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Shape,
    Data,
}

impl Origin {
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Shape => "shape",
            Origin::Data => "data",
        }
    }

    /// `Shape` only when both sides are `Shape` (§3.2, §6.1).
    pub fn join(a: Origin, b: Origin) -> Origin {
        if a == Origin::Shape && b == Origin::Shape {
            Origin::Shape
        } else {
            Origin::Data
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

/// The value of a scalar descriptor. `definition.md` §3.2 makes the term a
/// Boolean term when `τ = Bool`, so the two sorts are distinguished here
/// rather than by an `Option<TermId>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarTerm {
    Int(TermId),
    Bool(BoolId),
    /// No term: `Float`, `Complex`, `String`, `None` (see
    /// [`Dtype::carries_term`]).
    Opaque,
}

impl ScalarTerm {
    pub fn as_int(self) -> Option<TermId> {
        match self {
            ScalarTerm::Int(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_bool(self) -> Option<BoolId> {
        match self {
            ScalarTerm::Bool(b) => Some(b),
            _ => None,
        }
    }
}

/// `Scalar⟨τ, t, origin, defcond, ψ⟩` (`definition.md` §3.2).
#[derive(Debug, Clone)]
pub struct ScalarDesc {
    pub dtype: Dtype,
    pub term: ScalarTerm,
    pub origin: Origin,
    pub defcond: BoolId,
    pub psi: BoolId,
}

/// `Array⟨τ, r, dims, defcond, ψ⟩` (`definition.md` §3.3). The rank is
/// `dims.len()`, a compile-time constant.
#[derive(Debug, Clone)]
pub struct ArrayDesc {
    pub dtype: Dtype,
    pub dims: Vec<TermId>,
    pub defcond: BoolId,
    pub psi: BoolId,
}

impl ArrayDesc {
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeKind {
    List,
    Tuple,
}

impl CompositeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CompositeKind::List => "list",
            CompositeKind::Tuple => "tuple",
        }
    }
}

/// `Composite⟨kind, elems⟩` (`definition.md` §3.3). Every list mutation in
/// the baseline is unrolled, so the length is a compile-time constant and the
/// elements are descriptors of any kind.
#[derive(Debug, Clone)]
pub struct CompositeDesc {
    pub kind: CompositeKind,
    pub elems: Vec<Desc>,
}

/// The descriptor of one SSA value.
#[derive(Debug, Clone)]
pub enum Desc {
    Scalar(ScalarDesc),
    Array(ArrayDesc),
    Composite(CompositeDesc),
    /// A type object used as a value (`np.int64`, `Integer`), the baseline's
    /// `Value::Class`. It has no shape effect.
    Class(ZinniaType),
    /// Python `None`, the baseline's `Value::None`.
    None,
}

impl Desc {
    pub fn as_scalar(&self) -> Option<&ScalarDesc> {
        match self {
            Desc::Scalar(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&ArrayDesc> {
        match self {
            Desc::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_composite(&self) -> Option<&CompositeDesc> {
        match self {
            Desc::Composite(c) => Some(c),
            _ => None,
        }
    }

    pub fn dtype(&self) -> Option<Dtype> {
        match self {
            Desc::Scalar(s) => Some(s.dtype),
            Desc::Array(a) => Some(a.dtype),
            _ => None,
        }
    }

    /// The dimension vector for broadcasting: a scalar is rank 0, i.e. the
    /// empty vector (§5.3 equalises ranks by prepending 1s).
    pub fn dims(&self) -> Option<&[TermId]> {
        match self {
            Desc::Array(a) => Some(&a.dims),
            Desc::Scalar(_) => Some(&[]),
            _ => None,
        }
    }

    pub fn defcond(&self) -> Option<BoolId> {
        match self {
            Desc::Scalar(s) => Some(s.defcond),
            Desc::Array(a) => Some(a.defcond),
            _ => None,
        }
    }

    pub fn psi(&self) -> Option<BoolId> {
        match self {
            Desc::Scalar(s) => Some(s.psi),
            Desc::Array(a) => Some(a.psi),
            _ => None,
        }
    }

    /// A short name for the descriptor kind, used in rejection messages and
    /// in the JSON export.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Desc::Scalar(_) => "scalar",
            Desc::Array(_) => "array",
            Desc::Composite(_) => "composite",
            Desc::Class(_) => "class",
            Desc::None => "none",
        }
    }
}

// ---------------------------------------------------------------------------
// Value keys (`acceptance.md` §2.2)
// ---------------------------------------------------------------------------

/// Which arm of a two-way branch a block belongs to.
///
/// `S-If` (`definition.md` §6.1) and `E-CondExp` (§5.11) both evaluate two
/// arms under complementary path conditions. Without a component naming the
/// arm, statement `j` of the `then` arm and statement `j` of the `else` arm
/// would get the same [`ValueKey`] and the second would overwrite the first
/// in $\Delta$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arm {
    Then,
    Else,
}

impl Arm {
    /// The word the canonical key string uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Arm::Then => "then",
            Arm::Else => "else",
        }
    }
}

/// One component of the statement path of a [`ValueKey`].
///
/// A block contributes the index of the statement in it; entering the arm of
/// an `if` or of a conditional expression contributes the arm instead, so
/// that the two arms are told apart (`implementation.md` §5a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PathStep {
    /// The index of the enclosing statement in its block.
    Stmt(u32),
    /// Which arm of the enclosing two-way branch the block is.
    Arm(Arm),
}

impl PathStep {
    /// The component's spelling in [`ValueKey::encode`]: a decimal number for
    /// a statement, `then` or `else` for an arm. A Python identifier is
    /// neither, so no name can be mistaken for a component.
    pub fn as_string(self) -> String {
        match self {
            PathStep::Stmt(index) => index.to_string(),
            PathStep::Arm(arm) => arm.as_str().to_string(),
        }
    }
}

/// One inlined call on the way to a value. `callee` is the chip name;
/// `site` is the expression index of the call node inside the *calling*
/// statement, which is what tells two call sites of the same chip in one
/// statement apart.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CallSite {
    pub callee: String,
    pub site: u32,
}

/// The name phase 0 gives an SSA value.
///
/// `acceptance.md` §2.2 fixes the four components:
///
/// ```text
/// key = (statement path, iteration counters, call depth, expression index)
/// statement path     indices of the statement in each enclosing block, root first
/// iteration counters one counter per enclosing unrolled loop, innermost last
/// call depth         chip inlining depth, with the callee name
/// expression index   preorder index of the expression node inside its statement
/// ```
///
/// Read precisely, as [`KeyCursor`] implements it:
///
/// - `stmt_path` is the index of the enclosing statement in each enclosing
///   block, root first, with a [`PathStep::Arm`] component wherever the block
///   is one arm of an `if` or of a conditional expression. An inlined chip
///   body is a block like any other, so the path continues into the callee.
/// - `iters` has one entry per enclosing unrolled loop — `for` over `range`,
///   `for` over elements, a bounded `for` over a dynamic array, an unrolled
///   `while`, and a statically expanded comprehension — innermost last.
/// - `calls` has one [`CallSite`] per enclosing inlined chip call,
///   outermost first; its length is the call depth. §2.2 names only the
///   callee; the call site's expression index is added so that two calls of
///   the same chip within one statement get different keys.
/// - `expr_index` is the preorder index of the expression node within its
///   statement, counting from 0 and restarting at every statement and at
///   every inlined call.
///
/// Two traversals of the same AST with the same unrolling produce the same
/// keys. [`ValueKey::encode`] is the canonical string form that the JSON
/// export and `testing/phase0/refeval.py` share.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueKey {
    pub stmt_path: Vec<PathStep>,
    pub iters: Vec<u32>,
    pub calls: Vec<CallSite>,
    pub expr_index: u32,
}

impl ValueKey {
    /// The canonical string form, which the JSON export and
    /// `testing/phase0/refeval.py` share:
    ///
    /// ```text
    /// "<stmt path>|<iterations>|<calls>|<expression index>"
    /// ```
    ///
    /// The statement path is dot-joined [`PathStep`]s — a decimal statement
    /// index, or the word `then` / `else` for the arm of an `if` or of a
    /// conditional expression — so `"3.then.0|||5"` is expression 5 of
    /// statement 0 of the `then` arm of statement 3. The iterations are
    /// dot-joined decimal numbers, the calls dot-joined `callee@site` pairs,
    /// and any of the first three fields may be empty. A Python identifier
    /// contains none of `.`, `|` and `@`, so the encoding is unambiguous.
    pub fn encode(&self) -> String {
        let path = self
            .stmt_path
            .iter()
            .map(|step| step.as_string())
            .collect::<Vec<_>>()
            .join(".");
        let iters = self
            .iters
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(".");
        let calls = self
            .calls
            .iter()
            .map(|c| format!("{}@{}", c.callee, c.site))
            .collect::<Vec<_>>()
            .join(".");
        format!("{}|{}|{}|{}", path, iters, calls, self.expr_index)
    }

    /// The key of element `index` of a composite value.
    ///
    /// `E-Input-Composite` (§5.1) builds one descriptor per element of a
    /// `List` / `Tuple` input, and each element that is a data scalar takes a
    /// variable of its own; without a key per element the variables would
    /// share the composite's key and $\theta(u)$ could not tell them apart
    /// (`acceptance.md` §2.3). The position joins `iters`, which is the
    /// component this key indexes into, and the statement path stays empty:
    /// an expression always lies inside a statement, so no expression key can
    /// collide with an input's. `testing/phase0/refeval.py` records the
    /// element values under the same keys.
    pub fn component(&self, index: u32) -> ValueKey {
        let mut key = self.clone();
        key.iters.push(index);
        key
    }

    /// The structured form, for consumers that prefer not to parse
    /// [`ValueKey::encode`].
    pub fn to_json(&self) -> serde_json::Value {
        let calls: Vec<serde_json::Value> = self
            .calls
            .iter()
            .map(|c| serde_json::json!({"callee": c.callee, "site": c.site}))
            .collect();
        let stmt_path: Vec<String> = self.stmt_path.iter().map(|s| s.as_string()).collect();
        serde_json::json!({
            "key": self.encode(),
            "stmt_path": stmt_path,
            "iters": self.iters,
            "calls": calls,
            "expr_index": self.expr_index,
        })
    }
}

/// Saved state to restore when leaving a block, a loop or an inlined call.
/// Returned by the `enter_*` methods and consumed by the matching `exit_*`.
#[derive(Debug, Clone, Copy)]
pub struct KeyMark(u32);

/// Produces the [`ValueKey`]s of a traversal.
///
/// `crate::phase0::walk` drives it: `enter_block` /
/// `set_stmt` / `exit_block` around each block, `enter_loop` /
/// `set_iteration` / `exit_loop` around each unrolled loop, `enter_call` /
/// `exit_call` around each inlined chip body, and `next_expr` once per
/// expression node in preorder.
#[derive(Debug, Clone, Default)]
pub struct KeyCursor {
    stmt_path: Vec<PathStep>,
    iters: Vec<u32>,
    calls: Vec<CallSite>,
    next_expr: u32,
}

impl KeyCursor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enter a block of statements. The caller must call
    /// [`KeyCursor::set_stmt`] before evaluating each statement.
    pub fn enter_block(&mut self) -> KeyMark {
        self.stmt_path.push(PathStep::Stmt(0));
        KeyMark(self.next_expr)
    }

    /// Select the statement at `index` in the innermost block and restart
    /// the expression counter.
    pub fn set_stmt(&mut self, index: u32) {
        // `enter_block` pushed the slot; the walker never calls `set_stmt`
        // outside a block, so the innermost step is always a statement slot.
        if let Some(slot @ PathStep::Stmt(_)) = self.stmt_path.last_mut() {
            *slot = PathStep::Stmt(index);
        }
        self.next_expr = 0;
    }

    pub fn exit_block(&mut self, mark: KeyMark) {
        self.stmt_path.pop();
        self.next_expr = mark.0;
    }

    /// Enter one arm of an `if` (`S-If`, §6.1) or of a conditional
    /// expression (`E-CondExp`, §5.11).
    ///
    /// The arm joins the statement path and the expression counter restarts,
    /// so the two arms cannot collide and the keys after the branch do not
    /// depend on which arm a statically decided condition skipped.
    pub fn enter_arm(&mut self, arm: Arm) -> KeyMark {
        self.stmt_path.push(PathStep::Arm(arm));
        let mark = KeyMark(self.next_expr);
        self.next_expr = 0;
        mark
    }

    pub fn exit_arm(&mut self, mark: KeyMark) {
        self.stmt_path.pop();
        self.next_expr = mark.0;
    }

    pub fn enter_loop(&mut self) -> KeyMark {
        self.iters.push(0);
        KeyMark(self.next_expr)
    }

    /// Select iteration `index` of the innermost unrolled loop and restart
    /// the expression counter, so that the expressions a loop evaluates
    /// outside its body — a `while` guard, a comprehension's element — are
    /// keyed the same in every iteration and told apart by `iters`.
    pub fn set_iteration(&mut self, index: u32) {
        if let Some(slot) = self.iters.last_mut() {
            *slot = index;
        }
        self.next_expr = 0;
    }

    pub fn exit_loop(&mut self, mark: KeyMark) {
        self.iters.pop();
        self.next_expr = mark.0;
    }

    /// Enter an inlined chip body called from the expression at index
    /// `site` of the current statement.
    pub fn enter_call(&mut self, callee: &str, site: u32) -> KeyMark {
        self.calls.push(CallSite {
            callee: callee.to_string(),
            site,
        });
        let mark = KeyMark(self.next_expr);
        self.next_expr = 0;
        mark
    }

    pub fn exit_call(&mut self, mark: KeyMark) {
        self.calls.pop();
        self.next_expr = mark.0;
    }

    /// The key of the next expression node in preorder. Called exactly once
    /// per expression node, on entry, so the index is the preorder index.
    pub fn next_expr(&mut self) -> ValueKey {
        let key = self.at(self.next_expr);
        self.next_expr += 1;
        key
    }

    /// The key of the expression node at `expr_index` in the current
    /// statement, without advancing.
    pub fn at(&self, expr_index: u32) -> ValueKey {
        ValueKey {
            stmt_path: self.stmt_path.clone(),
            iters: self.iters.clone(),
            calls: self.calls.clone(),
            expr_index,
        }
    }
}
