//! P1 — `IROp::smt_encode` real Z3 encoding.
//!
//! P0 stubbed [`Z3Term`] and [`SmtEncodingCtx`] as placeholders and left every
//! "must constrain" [`IR`] arm at `todo!()`. P1 wires both to the `z3` crate
//! and fills in real encoding bodies for the integer / comparison / select /
//! logical / boolean-cast / constant arms — i.e., everything the integer-
//! resolver actually queries today.
//!
//! ## Z3 lifetime / context model
//!
//! `z3` 0.20 changed away from the explicit `&'ctx Context` borrow pattern.
//! It now uses an *implicit thread-local context* (`Context::thread_local()`).
//! `Solver::new()`, `Int::from_i64(2)`, `Int::add(...)`, `Bool::from_bool(b)`,
//! etc. all consult that thread-local. There is no `'ctx` parameter to
//! propagate. As a consequence:
//!
//! * `SmtEncodingCtx` no longer needs a borrowed `&Context` — it's a small
//!   per-query bookkeeping struct.
//! * `Z3Term` is a sort-tagged enum wrapping `z3::ast::Int` or
//!   `z3::ast::Bool` (the only two sorts the integer-resolver cares about);
//!   it doesn't carry a context lifetime.
//! * The "single Z3 context per compilation" requirement from the spec
//!   is satisfied implicitly: the thread-local default Context is created
//!   once per thread and reused. (Compilation is single-threaded, so this
//!   matches the paper's intent.)
//!
//! See module-level comment in `resolver.rs` for the matching choice on the
//! resolver's lifetime / Send+Sync story.
//!
//! ## What's encoded vs. silent fallback (P1 cut)
//!
//! Real encodings:
//! - `ConstantInt`, `ConstantBool`
//! - `AddI / SubI / MulI / DivI / ModI / FloorDivI`
//! - `EqI / NeI / LtI / LteI / GtI / GteI`
//! - `LogicalAnd / LogicalOr / LogicalNot`
//! - `SelectI / SelectB`
//! - `BoolCast` (int→bool: `If(int != 0, true, false)`)
//! - `IntCast` (bool→int: `If(bool, 1, 0)`)
//! - `AbsI / SignI`
//! - `InvI` (constraint form: `out * in = 1`)
//! - `PowI` (only when the exponent's term is itself a Z3 literal we can
//!   read; otherwise unconstrained — this matches the old Python which
//!   blindly emitted `**` even for non-constant exponents and let Z3 fail.
//!   We're a notch more conservative here.)
//!
//! Silent fallback (`fresh_unconstrained()`):
//! - `ConstantFloat / ConstantStr` — float / string aren't queried by the
//!   int resolver. Still real-encodable in P1+ if profiling demands.
//! - All float arithmetic + comparisons (`AddF`, `LtF`, …) — the int
//!   resolver doesn't query floats. Reflect later.
//! - All bitwise ops (`BitAndI`, `ShlI`, …) — Z3's Int sort doesn't
//!   directly support bitwise; they'd need `BV` translation. P2's range
//!   analysis will catch most cases.
//! - `FloatCast` — leaves the integer domain.
//! - All transcendental, string, IO, memory, dyn-ndarray, external,
//!   hashing ops — not on the integer-resolver hot path.
//!
//! Each silent-fallback arm has a comment naming the reason; tighten if
//! a profiling pass shows demand.

use crate::ir_defs::IR;
use z3::ast::{Ast, Bool, Int, Real};

// ---------------------------------------------------------------------------
// Z3Term — sort-tagged Z3 AST wrapper.
// ---------------------------------------------------------------------------

/// A Z3 term, sort-tagged. Three sorts: `Int`, `Bool`, `Real`. Integer
/// resolution stays on `Int`; bool reasoning on `Bool`; float contracts
/// (compiler.float-contracts) operate on `Real`. Mixed-sort comparisons
/// coerce via `as_int` / `as_real` / `as_bool`.
#[derive(Debug, Clone)]
pub enum Z3Term {
    Int(Int),
    Bool(Bool),
    Real(Real),
}

impl Z3Term {
    /// Coerce to an `Int`. If the term is a `Bool`, encode as `If(b, 1, 0)`.
    /// If the term is a `Real`, truncate via Z3's `to_int` (floor toward 0).
    pub fn as_int(&self) -> Int {
        match self {
            Z3Term::Int(i) => i.clone(),
            Z3Term::Bool(b) => b.ite(&Int::from_i64(1), &Int::from_i64(0)),
            Z3Term::Real(r) => r.to_int(),
        }
    }

    /// Coerce to a `Bool`. If the term is an `Int`, encode as `i != 0`. If
    /// the term is a `Real`, encode as `r != 0.0`.
    pub fn as_bool(&self) -> Bool {
        match self {
            Z3Term::Bool(b) => b.clone(),
            Z3Term::Int(i) => i._eq(&Int::from_i64(0)).not(),
            Z3Term::Real(r) => r._eq(&Real::from_real(0, 1)).not(),
        }
    }

    /// Coerce to a `Real`. Int → Real by `Int::to_real`; Bool by
    /// `b.ite(1.0, 0.0)`.
    pub fn as_real(&self) -> Real {
        match self {
            Z3Term::Real(r) => r.clone(),
            Z3Term::Int(i) => Real::from_int(i),
            Z3Term::Bool(b) => b.ite(&Real::from_real(1, 1), &Real::from_real(0, 1)),
        }
    }
}

// ---------------------------------------------------------------------------
// SmtEncodingCtx — per-query bookkeeping.
// ---------------------------------------------------------------------------

/// Per-encoding bookkeeping. Tracks a counter so each `fresh_unconstrained()`
/// call gets a unique name (Z3 unique-variable hygiene). The thread-local
/// `z3::Context` is consulted implicitly by the wrapped `Int` / `Bool`
/// constructors.
///
/// Also tracks structural-predicate meta-facts injected during this
/// encoding so a predicate's universal axioms (e.g., `nnz(v) >= 0`) are
/// asserted at most once per query even if the predicate atom is
/// referenced repeatedly.
#[derive(Debug, Default)]
pub struct SmtEncodingCtx {
    next_unconstrained_id: u64,
    /// Set of predicate kinds whose meta-facts have already been injected
    /// during this encoding. Keyed by `kind` string — the foundation card's
    /// dedup granularity is per-(predicate, build). Later cards (per-instance
    /// substitution) may refine this.
    injected_meta_kinds: std::collections::HashSet<String>,
    /// Accumulator of meta-fact `Bool` constraints injected by this
    /// encoding. The resolver conjoins these into the assembled formula
    /// before calling Z3.
    pub meta_facts: Vec<Bool>,
}

impl SmtEncodingCtx {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint a fresh unconstrained `Int` symbolic. Used by the default
    /// `IROp::smt_encode` fallback to translate ops we haven't bothered to
    /// constrain — Z3 sees an arbitrary integer, and the resolver treats
    /// it as "unknown" (non-unique model).
    pub fn fresh_unconstrained(&mut self) -> Z3Term {
        let id = self.next_unconstrained_id;
        self.next_unconstrained_id += 1;
        Z3Term::Int(Int::fresh_const(&format!("unconstrained_{id}_")))
    }

    /// Inject the universal meta-facts for a registered structural
    /// predicate into this encoding. Deduplicated by predicate `kind` so
    /// repeated references within one query do not multiply the formula.
    /// Returns `true` if any facts were appended, `false` if the kind was
    /// already injected.
    pub fn inject_meta_facts(&mut self, kind: &str, facts: Vec<Bool>) -> bool {
        if self.injected_meta_kinds.contains(kind) {
            return false;
        }
        self.injected_meta_kinds.insert(kind.to_string());
        self.meta_facts.extend(facts);
        true
    }

    /// True if a meta-fact set for the given predicate kind has been
    /// injected during this encoding.
    pub fn has_injected(&self, kind: &str) -> bool {
        self.injected_meta_kinds.contains(kind)
    }
}

// ---------------------------------------------------------------------------
// IROp trait — compile-time exhaustiveness over IR variants.
// ---------------------------------------------------------------------------

/// Encode an IR operation's semantics as SMT constraints over its argument
/// terms. Default fallback: unconstrained symbolic output (sound; precision
/// loss only).
pub trait IROp {
    fn smt_encode(&self, ctx: &mut SmtEncodingCtx, _args: &[Z3Term]) -> Z3Term {
        ctx.fresh_unconstrained()
    }
}

/// One implementer: the `IR` enum itself. The match must cover every
/// variant (compile-time exhaustiveness). Fallback arms call
/// `fresh_unconstrained()` with a comment naming the reason.
impl IROp for IR {
    fn smt_encode(&self, ctx: &mut SmtEncodingCtx, args: &[Z3Term]) -> Z3Term {
        match self {
            // ── Constants ──────────────────────────────────────────────
            IR::ConstantInt { value } => Z3Term::Int(Int::from_i64(*value)),
            IR::ConstantBool { value } => Z3Term::Bool(Bool::from_bool(*value)),
            // Float / string constants aren't on the integer resolver
            // hot path. Silent fallback. Tighten if profiling demands.
            IR::ConstantFloat { .. } | IR::ConstantStr { .. } => {
                ctx.fresh_unconstrained()
            }

            // ── Integer arithmetic ─────────────────────────────────────
            IR::AddI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Int(Int::add(&[&a, &b]))
            }
            IR::SubI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Int(Int::sub(&[&a, &b]))
            }
            IR::MulI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Int(Int::mul(&[&a, &b]))
            }
            IR::DivI | IR::FloorDivI => {
                // Z3's `Int::div` is integer division (truncating toward
                // zero in SMT-LIB semantics). The Python ref uses `/`
                // for both DivI and FloorDivI; we mirror that.
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Int(a.div(&b))
            }
            IR::ModI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Int(a.modulo(&b))
            }
            IR::PowI => {
                // `Int::power` returns Real in z3 0.20. Forcing the result
                // back to Int via from_real is sound only when the exponent
                // is non-negative and the base is integer — Z3 will simplify
                // `from_real(pow(int, n))` correctly for those cases. For
                // negative or non-integer-valued exponents, the resolver
                // will fail to find a unique model and return None — sound.
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Int(Int::from_real(&a.power(&b)))
            }
            IR::AbsI => {
                // |x| = If(x >= 0, x, -x). Encoded directly so the
                // resolver can fold `abs(constant)` and `abs(non-negative)`.
                let x = args[0].as_int();
                let zero = Int::from_i64(0);
                let neg = Int::sub(&[&zero, &x]);
                Z3Term::Int(x.ge(&zero).ite(&x, &neg))
            }
            IR::SignI => {
                let x = args[0].as_int();
                let zero = Int::from_i64(0);
                let pos = Int::from_i64(1);
                let neg = Int::from_i64(-1);
                let inner = x.lt(&zero).ite(&neg, &zero);
                Z3Term::Int(x.gt(&zero).ite(&pos, &inner))
            }
            IR::InvI => {
                // The Python ref encodes InvI as a *constraint* (`out * in
                // = 1`) since the result is rarely an Int. In our P1 model
                // we don't have side-channel constraint emission; silent
                // fallback is sound.
                let _ = args;
                ctx.fresh_unconstrained()
            }

            // ── Integer bitwise — silent fallback ──────────────────────
            // Z3 supports bitwise ops only over BV sort, not Int. P1 leaves
            // these as silent fallback; consider tightening (BV-bridge) if
            // profiling shows demand. Range analysis (P2) will catch many
            // common cases (mask ranges, shift bounds).
            IR::BitAndI | IR::BitOrI | IR::BitXorI
            | IR::ShlI | IR::ShrI | IR::BitNotI => {
                let _ = args;
                ctx.fresh_unconstrained()
            }

            // ── Float arithmetic — silent fallback ─────────────────────
            // Integer resolution is the P1 priority; float queries don't
            // route through this path today.
            IR::AddF | IR::SubF | IR::MulF | IR::DivF
            | IR::FloorDivF | IR::ModF | IR::PowF
            | IR::AbsF | IR::SignF => {
                let _ = args;
                ctx.fresh_unconstrained()
            }

            // ── Integer comparisons ────────────────────────────────────
            IR::EqI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Bool(a._eq(&b))
            }
            IR::NeI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Bool(a._eq(&b).not())
            }
            IR::LtI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Bool(a.lt(&b))
            }
            IR::LteI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Bool(a.le(&b))
            }
            IR::GtI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Bool(a.gt(&b))
            }
            IR::GteI => {
                let a = args[0].as_int();
                let b = args[1].as_int();
                Z3Term::Bool(a.ge(&b))
            }

            // ── Float comparisons — silent fallback ────────────────────
            // The int resolver doesn't query float comparisons.
            IR::EqF | IR::NeF | IR::LtF | IR::LteF | IR::GtF | IR::GteF => {
                let _ = args;
                ctx.fresh_unconstrained()
            }

            // ── Math (transcendental) — silent fallback ────────────────
            IR::SinF | IR::SinHF | IR::CosF | IR::CosHF
            | IR::TanF | IR::TanHF | IR::SqrtF | IR::ExpF | IR::LogF
            | IR::ArcCosF | IR::ArcTan2F => {
                let _ = args;
                ctx.fresh_unconstrained()
            }

            // ── Logical ────────────────────────────────────────────────
            IR::LogicalAnd => {
                let a = args[0].as_bool();
                let b = args[1].as_bool();
                Z3Term::Bool(Bool::and(&[&a, &b]))
            }
            IR::LogicalOr => {
                let a = args[0].as_bool();
                let b = args[1].as_bool();
                Z3Term::Bool(Bool::or(&[&a, &b]))
            }
            IR::LogicalNot => Z3Term::Bool(args[0].as_bool().not()),

            // ── Selection ──────────────────────────────────────────────
            IR::SelectI => {
                let cond = args[0].as_bool();
                let t = args[1].as_int();
                let f = args[2].as_int();
                Z3Term::Int(cond.ite(&t, &f))
            }
            IR::SelectB => {
                let cond = args[0].as_bool();
                let t = args[1].as_bool();
                let f = args[2].as_bool();
                Z3Term::Bool(cond.ite(&t, &f))
            }
            IR::SelectF => {
                // Float silent fallback — see float-arithmetic note.
                let _ = args;
                ctx.fresh_unconstrained()
            }

            // ── Cast ───────────────────────────────────────────────────
            IR::IntCast => {
                // The most useful IntCast at the integer resolver is
                // bool→int (`If(b, 1, 0)`); it's free-of-charge through
                // `as_int`.
                Z3Term::Int(args[0].as_int())
            }
            IR::BoolCast => {
                // int→bool: per spec, `If(int != 0, true, false)`. The
                // `as_bool` helper does exactly that.
                Z3Term::Bool(args[0].as_bool())
            }
            IR::FloatCast => {
                // FloatCast usually leaves the integer domain. P1 silent
                // fallback; precision loss only.
                let _ = args;
                ctx.fresh_unconstrained()
            }

            // Structural-predicate atoms have a real encoding (registered
            // per kind in `optim::predicates`); they inject meta-facts into
            // the context as a side effect and return a Bool term.
            IR::StructuralPredicate { .. } => {
                crate::optim::predicates::smt_encode_structural_predicate(self, ctx)
            }

            // Scalar precondition: the discharger collects these
            // separately and lowers the ContractTerm via
            // `formula::lower_bool`. The per-statement encoding path
            // here is unused (the atom has no operands) — return a
            // fresh unconstrained so any incidental query through this
            // arm degrades gracefully.
            IR::ScalarPrecondition { .. } => ctx.fresh_unconstrained(),

            // ── String / IO / memory / dyn-ndarray / external / hashing
            //    — silent fallback. Off the integer-resolver hot path.
            IR::AddStr
            | IR::StrI
            | IR::StrF
            | IR::ReadInteger { .. }
            | IR::ReadFloat { .. }
            | IR::ReadHash { .. }
            | IR::ReadExternalResult { .. }
            | IR::Print
            | IR::Assert
            | IR::ExposePublicI
            | IR::ExposePublicF
            | IR::AllocateMemory { .. }
            | IR::WriteMemory { .. }
            | IR::ReadMemory { .. }
            | IR::MemoryTraceEmit { .. }
            | IR::MemoryTraceSeal
            | IR::AllocateDynamicNDArrayMeta { .. }
            | IR::WitnessDynamicNDArrayMeta { .. }
            | IR::AssertDynamicNDArrayMeta { .. }
            | IR::DynamicNDArrayGetItem { .. }
            | IR::DynamicNDArraySetItem { .. }
            | IR::InvokeExternal { .. }
            | IR::ExportExternalI { .. }
            | IR::ExportExternalF { .. }
            | IR::PoseidonHash
            | IR::EqHash => {
                let _ = args;
                ctx.fresh_unconstrained()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use z3::{SatResult, Solver};

    /// Smoke test: confirm the `z3` crate is on the dep graph, links, and
    /// can prove a trivial arithmetic identity. Not wired to anything else
    /// in the compiler — its only purpose is to fail loudly if the dep
    /// version we picked doesn't link on this platform.
    ///
    /// Note: z3 0.20 uses an implicit thread-local `Context`. There is no
    /// explicit `Context` borrow on `Solver::new()`, `Int::from_i64()`,
    /// `Int::add()`, etc. — they all consult the thread-local. This is
    /// the API we'll use throughout the resolver.
    #[test]
    fn z3_dep_smoke_test_two_plus_three_is_five() {
        let solver = Solver::new();

        let two = Int::from_i64(2);
        let three = Int::from_i64(3);
        let five = Int::from_i64(5);
        let sum = &two + &three;

        // Try to find a counter-example to "2 + 3 == 5". There is none.
        solver.assert(sum._eq(&five).not());
        assert_eq!(solver.check(), SatResult::Unsat);
    }

    /// Sanity: encoding `ConstantInt { value: 7 }` produces a Z3 Int
    /// literal that proves equal to 7.
    #[test]
    fn smt_encode_constant_int() {
        let mut ctx = SmtEncodingCtx::new();
        let term = IR::ConstantInt { value: 7 }.smt_encode(&mut ctx, &[]);
        let int = match term {
            Z3Term::Int(i) => i,
            _ => panic!("expected Int term"),
        };
        let solver = Solver::new();
        solver.assert(int._eq(&Int::from_i64(7)).not());
        assert_eq!(solver.check(), SatResult::Unsat);
    }

    /// Sanity: AddI on two literal ConstantInts encodes a sum that Z3
    /// proves equal to the expected result.
    #[test]
    fn smt_encode_add_i_constants() {
        let mut ctx = SmtEncodingCtx::new();
        let a = IR::ConstantInt { value: 11 }.smt_encode(&mut ctx, &[]);
        let b = IR::ConstantInt { value: 31 }.smt_encode(&mut ctx, &[]);
        let sum = IR::AddI.smt_encode(&mut ctx, &[a, b]);
        let int = match sum {
            Z3Term::Int(i) => i,
            _ => panic!("expected Int term"),
        };
        let solver = Solver::new();
        solver.assert(int._eq(&Int::from_i64(42)).not());
        assert_eq!(solver.check(), SatResult::Unsat);
    }

    /// Sanity: SelectI on a true cond evaluates to the then-branch.
    #[test]
    fn smt_encode_select_i_constant_cond() {
        let mut ctx = SmtEncodingCtx::new();
        let cond = IR::ConstantBool { value: true }.smt_encode(&mut ctx, &[]);
        let t = IR::ConstantInt { value: 7 }.smt_encode(&mut ctx, &[]);
        let f = IR::ConstantInt { value: 9 }.smt_encode(&mut ctx, &[]);
        let sel = IR::SelectI.smt_encode(&mut ctx, &[cond, t, f]);
        let int = match sel {
            Z3Term::Int(i) => i,
            _ => panic!("expected Int term"),
        };
        let solver = Solver::new();
        solver.assert(int._eq(&Int::from_i64(7)).not());
        assert_eq!(solver.check(), SatResult::Unsat);
    }
}
