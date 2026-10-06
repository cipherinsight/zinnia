//! Store-to-load forwarding at constant memory addresses, and removal of
//! the stores no load observes.
//!
//! The frontend keeps an array in a memory segment when it is written under
//! a runtime condition in a loop (`result[i] = ...` in the branches of an
//! `if`): each write is `WriteMemory(seg, k, select(p, new, ReadMemory(seg,
//! k)))` at a constant address `k`, and the final value is read back with
//! `ReadMemory(seg, k)`. The value of such a read is a statement already in
//! the graph, but passes that reason about values (`select_merge`) cannot
//! see through memory. This pass makes the dataflow explicit.
//!
//! **Forwarding.** Walking the statements in order, the pass tracks for each
//! segment the statement last written at each constant address. A write at
//! a runtime address makes every slot of its segment unknown (it may hit
//! any), until a constant write sets a slot again. A `ReadMemory(seg, k)`
//! with a constant `k` inside the segment, whose slot holds a known
//! statement `v`, is replaced by `v` (its users read `v`) and removed. A
//! read of a slot nothing has written yet, in a segment allocated with the
//! initial value 0 (`np.zeros`, `np.empty`), is replaced by the constant 0
//! (the same field element for an integer 0 and a float 0.0).
//!
//! **Dead stores.** A constant write `WriteMemory(seg, k, v)` is removed when
//! no remaining access can observe it: walking backwards, after it and
//! before the next constant write at `k` there is no remaining read at `k`,
//! no read at a runtime address of `seg`, and no other operation on `seg`
//! (memory traces, dynamic-array accesses). A store is kept when nothing
//! overwrites it before the end (dropping writes no one reads is
//! `UnreadMemoryEliminator`'s job, with its own argument).
//!
//! **Soundness.** In the IR's semantics (the mock: `read_memory` returns the
//! slot, `write_memory` sets it, `allocate_memory` fills every slot with
//! the initial value) a read at a constant in-range address `k` returns the
//! last value written at `k`, or the initial value when no write has hit
//! `k`; when every write to the segment since the allocation is at a
//! constant address, which slots were written is known, and the value is
//! `v`'s (or the initial 0). So forwarding preserves every value. A read at a constant address
//! inside its segment has no constraint of its own on any backend (the
//! address is a compile-time slot), so removing it removes no check. A
//! removed store is overwritten before any remaining access could read it,
//! so every remaining read returns the same value as before. No assertion is
//! added or removed.
//!
//! **Cost.** Never more: reads and writes at constant addresses only
//! disappear. The gain is indirect: the values the dead stores wrote and
//! the selects that merged old and new slot values become unused (and
//! dead-code elimination removes them) once the reads that kept them are
//! forwarded.

use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct ConstantMemoryForwarding;

impl IRPass for ConstantMemoryForwarding {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let forwarded = forwarded_reads(&graph);
        let dead = dead_stores(&graph, &forwarded);
        if forwarded.is_empty() && dead.is_empty() {
            return graph;
        }
        let mut b = IRBuilder::new();
        let mut lookup: HashMap<StmtId, Value> = HashMap::new();
        for stmt in &graph.stmts {
            if let Some(&slot) = forwarded.get(&stmt.stmt_id) {
                let value = match slot {
                    Slot::Written(value) => lookup[&value].clone(),
                    Slot::Zero => b.ir_constant_int(0),
                };
                lookup.insert(stmt.stmt_id, value);
                continue;
            }
            if dead.contains(&stmt.stmt_id) {
                lookup.insert(stmt.stmt_id, Value::None);
                continue;
            }
            let args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
            let value = b.create_ir(&stmt.ir, &args);
            lookup.insert(stmt.stmt_id, value);
        }
        let mut out = b.export_ir_graph();
        out.resolver_mut().on_ir_mutated(&[]);
        out
    }
}

fn constant_of(graph: &IRGraph, s: StmtId) -> Option<i64> {
    match graph.stmts[s as usize].ir {
        IR::ConstantInt { value } => Some(value),
        _ => None,
    }
}

/// The memory segment a statement touches, other than through
/// `ReadMemory` / `WriteMemory` (those are handled by address).
fn other_access(ir: &IR) -> Option<u32> {
    match *ir {
        IR::MemoryTraceEmit { segment_id, .. }
        // A table read may read any slot (and its segment is not written
        // after it).
        | IR::ReadTable { segment_id }
        | IR::DynamicNDArrayGetItem { segment_id, .. }
        | IR::DynamicNDArraySetItem { segment_id, .. } => Some(segment_id),
        _ => None,
    }
}

/// What a forwarded read returns.
#[derive(Clone, Copy)]
enum Slot {
    /// The value of this statement (the last one written).
    Written(StmtId),
    /// The segment's initial value 0 (nothing written yet).
    Zero,
}

/// What is known about one segment's slots at a point of the walk.
struct Segment {
    size: i64,
    /// The statement last written at each constant address.
    written: HashMap<i64, StmtId>,
    /// Every write so far was at a constant address, so an address not in
    /// `written` still holds the initial value.
    only_constant_writes: bool,
    zero_initialized: bool,
}

/// `read -> slot` for every read at a constant in-range address whose slot
/// is known (a written value is chained through forwarded reads to the
/// statement first written).
fn forwarded_reads(graph: &IRGraph) -> HashMap<StmtId, Slot> {
    let mut segments: HashMap<u32, Segment> = HashMap::new();
    let mut forwarded: HashMap<StmtId, Slot> = HashMap::new();
    let forget = |segment: &mut Segment| {
        segment.written.clear();
        segment.only_constant_writes = false;
    };
    for stmt in &graph.stmts {
        match (&stmt.ir, stmt.arguments.as_slice()) {
            (IR::AllocateMemory { segment_id, size, init_value }, _) => {
                let segment = Segment {
                    size: i64::from(*size),
                    written: HashMap::new(),
                    only_constant_writes: true,
                    zero_initialized: *init_value == 0,
                };
                segments.insert(*segment_id, segment);
            }
            (IR::WriteMemory { segment_id }, &[addr, value]) => {
                let Some(segment) = segments.get_mut(segment_id) else { continue };
                match constant_of(graph, addr) {
                    Some(k) => {
                        let value = match forwarded.get(&value) {
                            Some(Slot::Written(source)) => *source,
                            _ => value,
                        };
                        segment.written.insert(k, value);
                    }
                    None => forget(segment),
                }
            }
            (IR::ReadMemory { segment_id }, &[addr]) => {
                let (Some(segment), Some(k)) = (segments.get(segment_id), constant_of(graph, addr)) else { continue };
                if !(0 <= k && k < segment.size) {
                    continue;
                }
                let slot = match segment.written.get(&k) {
                    Some(&value) => Some(Slot::Written(value)),
                    None if segment.only_constant_writes && segment.zero_initialized => Some(Slot::Zero),
                    None => None,
                };
                if let Some(slot) = slot {
                    forwarded.insert(stmt.stmt_id, slot);
                }
            }
            (ir, _) => {
                if let Some(segment) = other_access(ir).and_then(|id| segments.get_mut(&id)) {
                    forget(segment);
                }
            }
        }
    }
    forwarded
}

/// The constant writes no remaining access observes before a later
/// constant write at the same address overwrites them. A backward walk
/// keeps, per segment, the addresses a later constant write overwrites
/// before any remaining access reads them (none at the end: a store nothing
/// overwrites is kept). A remaining read at `k` removes `k`; a read at a
/// runtime address or another access to the segment removes every address;
/// a constant write at `k` is dead when `k` is in the set, and then adds
/// `k`; a write at a runtime address changes nothing (it may miss `k`).
fn dead_stores(graph: &IRGraph, forwarded: &HashMap<StmtId, Slot>) -> HashSet<StmtId> {
    let mut overwritten: HashMap<u32, HashSet<i64>> = HashMap::new();
    let mut dead = HashSet::new();
    for stmt in graph.stmts.iter().rev() {
        match (&stmt.ir, stmt.arguments.as_slice()) {
            (IR::ReadMemory { segment_id }, &[addr]) if !forwarded.contains_key(&stmt.stmt_id) => {
                let segment = overwritten.entry(*segment_id).or_default();
                match constant_of(graph, addr) {
                    Some(k) => {
                        segment.remove(&k);
                    }
                    None => segment.clear(),
                }
            }
            (IR::WriteMemory { segment_id }, &[addr, _]) => {
                let Some(k) = constant_of(graph, addr) else { continue };
                let segment = overwritten.entry(*segment_id).or_default();
                if !segment.insert(k) {
                    dead.insert(stmt.stmt_id);
                }
            }
            (ir, _) => {
                if let Some(segment) = other_access(ir) {
                    overwritten.entry(segment).or_default().clear();
                }
            }
        }
    }
    dead
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
    use crate::prove::types::ProvingParams;

    /// The mock's verdict and the exposed values in statement order.
    fn run(graph: &IRGraph, inputs: &[(&str, i64)]) -> (bool, Vec<String>) {
        let mut circuit_inputs = CircuitInputs::new();
        for (name, value) in inputs {
            circuit_inputs.params.push(InputParam {
                name: name.to_string(),
                is_public: false,
                dtype: serde_json::Value::Null,
                value: InputNode::Int(*value),
            });
        }
        let witness = ResolvedWitness::new(circuit_inputs, ProvingParams::default().precision_bits);
        let mut synth = crate::prove::mock::synthesizer::MockSynthesizer::new(witness, ProvingParams::default());
        crate::prove::interpreter::interpret_ir(graph, &mut synth).expect("mock run");
        let mut outputs: Vec<(u32, String)> = synth
            .public_outputs
            .iter()
            .map(|(label, value)| (label.rsplit('_').next().unwrap().parse().unwrap(), value.clone()))
            .collect();
        outputs.sort();
        (synth.satisfied, outputs.into_iter().map(|(_, v)| v).collect())
    }

    fn count(graph: &IRGraph, f: fn(&IR) -> bool) -> usize {
        graph.stmts.iter().filter(|s| f(&s.ir)).count()
    }

    fn input(b: &mut IRBuilder, name: &str) -> Value {
        b.ir_read_integer(InputPath::new(name, vec![]), false)
    }

    /// A zero-initialized segment of 3: `m[0] = x` under `p = x < y` (as
    /// `select(p, x, m[0])`), then `m[0] = y` under not `p`, then `m[i] = 7`
    /// at the runtime address `i` when `dynamic`, then `m[0]`, `m[1]` read
    /// and exposed.
    fn conditional_writes(dynamic: bool) -> IRGraph {
        let mut b = IRBuilder::new();
        let (x, y, i) = (input(&mut b, "x"), input(&mut b, "y"), input(&mut b, "i"));
        let segment = 0;
        b.ir_allocate_memory(segment, 3, 0);
        let zero = b.ir_constant_int(0);
        let p = b.ir_less_than_i(&x, &y);
        let old = b.ir_read_memory(segment, &zero);
        let first = b.ir_select_i(&p, &x, &old);
        b.ir_write_memory(segment, &zero, &first);
        let not_p = b.ir_logical_not(&p);
        let old = b.ir_read_memory(segment, &zero);
        let second = b.ir_select_i(&not_p, &y, &old);
        b.ir_write_memory(segment, &zero, &second);
        if dynamic {
            let three = b.ir_constant_int(3);
            b.ir_assert_u128_lt(&i, &three);
            let seven = b.ir_constant_int(7);
            b.ir_write_memory(segment, &i, &seven);
        }
        let one = b.ir_constant_int(1);
        for addr in [&zero, &one] {
            let v = b.ir_read_memory(segment, addr);
            b.ir_expose_public_i(&v);
        }
        b.export_ir_graph()
    }

    #[test]
    fn constant_reads_are_forwarded_and_overwritten_stores_dropped() {
        let graph = conditional_writes(false);
        let out = ConstantMemoryForwarding.exec(graph.clone());
        // Every read is forwarded (the unwritten m[1] to the initial 0); the
        // first store is overwritten before any remaining read.
        assert_eq!(count(&out, |ir| matches!(ir, IR::ReadMemory { .. })), 0);
        assert_eq!(count(&out, |ir| matches!(ir, IR::WriteMemory { .. })), 1);
        for (x, y) in [(1, 2), (2, 1), (3, 3), (-4, 5)] {
            let inputs = [("x", x), ("y", y), ("i", 0)];
            assert_eq!(run(&out, &inputs), run(&graph, &inputs), "x={x} y={y}");
        }
    }

    #[test]
    fn a_runtime_write_stops_forwarding_its_segment() {
        let graph = conditional_writes(true);
        let out = ConstantMemoryForwarding.exec(graph.clone());
        // The reads after the runtime write stay (it may hit either slot);
        // the stores before it are kept for them except the overwritten one.
        assert_eq!(count(&out, |ir| matches!(ir, IR::ReadMemory { .. })), 2);
        assert_eq!(count(&out, |ir| matches!(ir, IR::WriteMemory { .. })), 2);
        // Every address the write may take (the mock has no value for a
        // write outside the segment).
        for i in 0..3 {
            for (x, y) in [(1, 2), (2, 1)] {
                let inputs = [("x", x), ("y", y), ("i", i)];
                assert_eq!(run(&out, &inputs), run(&graph, &inputs), "x={x} y={y} i={i}");
            }
        }
    }
}
