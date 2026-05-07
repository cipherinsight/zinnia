//! P0 — `IROp::smt_encode` trait skeleton.
//!
//! This module forward-declares the surface that P1's SMT resolver will
//! plug into: a trait method [`IROp::smt_encode`] that translates one IR
//! operation into a Z3 term over its argument terms. P0 ships:
//!
//! * Lightweight placeholder types [`Z3Term`] and [`SmtEncodingCtx`]. Their
//!   bodies are empty / `todo!()` — P1 fills them in along with the `z3`
//!   crate dependency.
//!
//! * A default trait fallback that calls [`SmtEncodingCtx::fresh_unconstrained`]
//!   (also `todo!()` for now). The default is conservative: an op that
//!   doesn't override `smt_encode` produces an unconstrained symbolic
//!   output, which P1's resolver treats as "unknown" and falls back to
//!   the non-SMT path. Soundness preserved; only precision lost.
//!
//! * Explicit `todo!()` impls for the IR ops that *should* constrain
//!   their output (arithmetic, comparison, select, cast). The point is
//!   compile-time exhaustiveness: a future contributor adding a new op
//!   to one of these categories has to either give it a real encoding
//!   or accept the fallback explicitly. `todo!()` panics at runtime if
//!   anyone actually invokes this in P0; that's intended — the trait
//!   is not wired up yet.

use crate::ir_defs::IR;

// ---------------------------------------------------------------------------
// Forward-declared placeholder types — P1 fills these in.
// ---------------------------------------------------------------------------

/// A Z3 term placeholder. P1 replaces this with the real `z3::ast::Dynamic`
/// (or a thin wrapper) once the `z3` crate is on the dep graph.
#[derive(Debug, Clone)]
pub struct Z3Term;

/// Per-compilation SMT encoding context. P1 will hold the Z3 context, the
/// per-ptr cache, and the timeout budget here.
#[derive(Debug, Default)]
pub struct SmtEncodingCtx;

impl SmtEncodingCtx {
    pub fn new() -> Self {
        Self
    }

    /// Mint a fresh symbolic term with no constraints attached. Used by the
    /// default `IROp::smt_encode` fallback; P1 fills in.
    pub fn fresh_unconstrained(&mut self) -> Z3Term {
        // P1 fills in the body. Default fallback is unconstrained on
        // purpose — see module-level comment.
        todo!("SmtEncodingCtx::fresh_unconstrained: P1 wires this to z3")
    }
}

// ---------------------------------------------------------------------------
// IROp trait — compile-time exhaustiveness over IR variants.
// ---------------------------------------------------------------------------

/// Encode an IR operation's semantics as SMT constraints over its argument
/// terms. P0 defines the surface; P1 fills in the bodies and the resolver
/// that consumes them.
pub trait IROp {
    /// Encode this op's semantics. Default: unconstrained symbolic output.
    fn smt_encode(&self, ctx: &mut SmtEncodingCtx, _args: &[Z3Term]) -> Z3Term {
        ctx.fresh_unconstrained()
    }
}

/// The implementer of `IROp` is the IR enum itself — one trait, one big
/// dispatch. Today's body groups "must constrain" ops as explicit `todo!()`
/// arms so future contributors are forced to either add a real encoding or
/// migrate the op into the silent-fallback set.
///
/// **P0 contract**: do not call `IR::smt_encode` from anywhere yet. The
/// `todo!()` arms exist purely to fail loudly if a P1+ rewrite tries to
/// invoke them before filling them in.
impl IROp for IR {
    fn smt_encode(&self, ctx: &mut SmtEncodingCtx, args: &[Z3Term]) -> Z3Term {
        match self {
            // ── Constants — P1 fills in (literal embedding) ────────────
            IR::ConstantInt { .. }
            | IR::ConstantFloat { .. }
            | IR::ConstantBool { .. }
            | IR::ConstantStr { .. } => {
                // P1 fills in the body; the impl exists here to prevent
                // silent fallback for these ops.
                todo!("IR::smt_encode: constants — P1")
            }

            // ── Integer arithmetic — must constrain ────────────────────
            IR::AddI
            | IR::SubI
            | IR::MulI
            | IR::DivI
            | IR::FloorDivI
            | IR::ModI
            | IR::PowI
            | IR::AbsI
            | IR::SignI
            | IR::InvI
            | IR::BitAndI
            | IR::BitOrI
            | IR::BitXorI
            | IR::ShlI
            | IR::ShrI
            | IR::BitNotI => {
                // P1 fills in the body; the impl exists here to prevent
                // silent fallback for these ops.
                todo!("IR::smt_encode: integer arithmetic — P1")
            }

            // ── Float arithmetic — must constrain ──────────────────────
            IR::AddF
            | IR::SubF
            | IR::MulF
            | IR::DivF
            | IR::FloorDivF
            | IR::ModF
            | IR::PowF
            | IR::AbsF
            | IR::SignF => {
                // P1 fills in the body; the impl exists here to prevent
                // silent fallback for these ops.
                todo!("IR::smt_encode: float arithmetic — P1")
            }

            // ── Comparisons — must constrain ───────────────────────────
            IR::EqI
            | IR::NeI
            | IR::LtI
            | IR::LteI
            | IR::GtI
            | IR::GteI
            | IR::EqF
            | IR::NeF
            | IR::LtF
            | IR::LteF
            | IR::GtF
            | IR::GteF => {
                // P1 fills in the body; the impl exists here to prevent
                // silent fallback for these ops.
                todo!("IR::smt_encode: comparisons — P1")
            }

            // ── Selection — must constrain ─────────────────────────────
            IR::SelectI | IR::SelectF | IR::SelectB => {
                // P1 fills in the body; the impl exists here to prevent
                // silent fallback for these ops.
                todo!("IR::smt_encode: select — P1")
            }

            // ── Cast — must constrain ──────────────────────────────────
            IR::IntCast | IR::FloatCast | IR::BoolCast => {
                // P1 fills in the body; the impl exists here to prevent
                // silent fallback for these ops.
                todo!("IR::smt_encode: cast — P1")
            }

            // ── Logical — must constrain ───────────────────────────────
            IR::LogicalAnd | IR::LogicalOr | IR::LogicalNot => {
                // P1 fills in the body; the impl exists here to prevent
                // silent fallback for these ops.
                todo!("IR::smt_encode: logical — P1")
            }

            // ── Math (transcendental) — silent fallback OK ─────────────
            // These are not amenable to integer-arithmetic SMT in the
            // common case; treating them as unconstrained symbolic is
            // sound (precision loss only).
            IR::SinF
            | IR::SinHF
            | IR::CosF
            | IR::CosHF
            | IR::TanF
            | IR::TanHF
            | IR::SqrtF
            | IR::ExpF
            | IR::LogF => ctx.fresh_unconstrained(),

            // ── String, IO, memory, dynamic-ndarray, externals, hashing
            //    — silent fallback. These don't produce values that the
            //    integer-resolver would query. Falling through to
            //    unconstrained is the right default; P1 may revisit on
            //    a per-op basis if a use case appears.
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
                let _ = args; // suppress unused warning
                ctx.fresh_unconstrained()
            }
        }
    }
}
