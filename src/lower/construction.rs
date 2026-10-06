//! Constructions: the ways to lower an operator, and the registry.
//!
//! A [`Construction`] looks at one group (and the facts and the rest of the
//! graph through [`Context`]) and offers [`Candidate`]s: implementation
//! nodes with preconditions, postconditions, a cost on the compile target
//! and an emitter. It keys on the operator, its shapes and facts only, never
//! on a program.
//!
//! Every construction's doc comment states its preconditions,
//! postconditions, cost formula per target, and its soundness and
//! completeness argument: on inputs satisfying the preconditions its
//! constraints are satisfied exactly by NumPy's result, and every prover
//! hint is pinned to one value.

use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;

use crate::builder::IRBuilder;
use crate::ir::{IRGraph, IRStatement};
use crate::types::{StmtId, Value};

use super::emit::EmitCx;
use super::facts::FreeFacts;
use super::graph::{Fact, Group, GroupId, Lane, Requirement};

/// The backend a program is compiled for (`ZinniaConfig.backend`). Any
/// lowered IR proves on any backend; the target only selects costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// halo2-base: costs in advice + lookup cells.
    Halo2Base,
    /// Noir: costs in `bb gates`.
    Noir,
}

impl Target {
    /// The target of a `ZinniaConfig.backend` name: `noir`, else halo2-base
    /// (the default backend, `halo2`).
    pub fn from_backend(backend: &str) -> Target {
        if backend == "noir" {
            Target::Noir
        } else {
            Target::Halo2Base
        }
    }
}

/// One way to lower an operator. Implementations are stateless values
/// listed in [`super::registry`].
pub trait Construction: Sync {
    /// A short, stable name (logs, tests, tie-breaks).
    fn name(&self) -> &'static str;

    /// The candidates this construction offers for `group`; none when it
    /// does not apply. A candidate may also produce lanes of other groups
    /// (a sort carrying a gathered payload), found through `cx`.
    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate>;
}

/// An implementation node a construction offers.
pub struct Candidate {
    /// The requirements it consumes: shared values and facts it relies on
    /// (each must be produced by another selected node, be a free fact, or
    /// be emitted as a check). Its operands need not be listed.
    pub pre: Vec<Requirement>,
    /// The requirements it establishes: lanes, shared values, and facts its
    /// constraints prove.
    pub post: Vec<Requirement>,
    /// Cost in the target's metric.
    pub cost: u64,
    /// Emits the node's IR.
    pub emitter: Box<dyn Emitter>,
}

/// Emits one selected implementation node.
///
/// The driver calls `begin` once, before anything else of the node, at the
/// point of the program where its first lane is needed; the shared operands
/// of every group whose lanes it produces are lowered by then. It then calls
/// `lane` for each lane in its postconditions, where that lane's placeholder
/// stands (its own operands are lowered), and `shared` for each shared
/// value a consumer asks for.
pub trait Emitter {
    fn begin(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx) {}

    /// The value of `lane`.
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value;

    /// The shared value `name` of group `group` (downcast by its consumer).
    fn shared(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, group: GroupId, name: &'static str) -> Rc<dyn Any> {
        unreachable!("this node produces no shared value {name} of group {group}")
    }
}

/// What a construction may look at while offering candidates.
pub struct Context<'a> {
    pub target: Target,
    pub graph: &'a IRGraph,
    pub groups: &'a [Group],
    pub(super) facts: &'a FreeFacts<'a>,
    pub(super) users: &'a HashMap<StmtId, Vec<StmtId>>,
    pub(super) lane_group: &'a HashMap<StmtId, GroupId>,
}

impl Context<'_> {
    /// `true` when `fact` holds without any construction: proved by the
    /// range / SMT engine, or established by a statement the program emits
    /// anyway (a scalar gadget's postcondition).
    pub fn proves(&self, fact: &Fact) -> bool {
        self.facts.proves(fact)
    }

    pub fn stmt(&self, id: StmtId) -> &IRStatement {
        &self.graph.stmts[id as usize]
    }

    /// The statements that read `id`'s value.
    pub fn users(&self, id: StmtId) -> &[StmtId] {
        self.users.get(&id).map_or(&[], Vec::as_slice)
    }

    /// The group whose lane `stmt` is.
    pub fn group_of_lane(&self, stmt: StmtId) -> Option<&Group> {
        self.lane_group.get(&stmt).map(|&g| &self.groups[g])
    }
}
