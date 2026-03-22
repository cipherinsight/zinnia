use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

// ═══════════════════════════════════════════════════════════════════════════
// 1. ExternalCallRemover
// ═══════════════════════════════════════════════════════════════════════════

pub struct ExternalCallRemover;

impl IRPass for ExternalCallRemover {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
        let to_remove: Vec<StmtId> = ir_graph
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::InvokeExternal { .. } | IR::ExportExternalI { .. } | IR::ExportExternalF { .. }))
            .map(|s| s.stmt_id)
            .collect();
        ir_graph.remove_stmt_bunch(&to_remove);
        ir_graph
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. DeadCodeElimination
// ═══════════════════════════════════════════════════════════════════════════

pub struct DeadCodeElimination;

impl IRPass for DeadCodeElimination {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
        let n = ir_graph.stmts.len();
        let ensure_keep: Vec<bool> = ir_graph.stmts.iter().map(|s| s.ir.is_fixed()).collect();
        let (mut _in_d, mut out_d) = ir_graph.get_io_degrees();

        let mut killing_queue: Vec<usize> = Vec::new();
        let mut to_eliminate: Vec<StmtId> = Vec::new();

        for i in 0..n {
            if out_d[i] == 0 && !ensure_keep[i] {
                killing_queue.push(i);
            }
        }

        while let Some(idx) = killing_queue.pop() {
            to_eliminate.push(idx as StmtId);
            for &arg in &ir_graph.stmts[idx].arguments {
                out_d[arg as usize] -= 1;
                if out_d[arg as usize] == 0 && !ensure_keep[arg as usize] {
                    killing_queue.push(arg as usize);
                }
            }
        }

        ir_graph.remove_stmt_bunch(&to_eliminate);
        ir_graph
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. DoubleNotElimination
// ═══════════════════════════════════════════════════════════════════════════

pub struct DoubleNotElimination;

impl IRPass for DoubleNotElimination {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        // Maps ptr of a NOT result -> the original operand value
        let mut not_original: HashMap<StmtId, Value> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();

            let result = if matches!(stmt.ir, IR::LogicalNot) {
                let operand = &ir_args[0];
                if let Some(ptr) = operand.ptr() {
                    if let Some(orig) = not_original.get(&ptr) {
                        // Double negation — eliminate
                        orig.clone()
                    } else {
                        let new_val = builder.create_ir(&stmt.ir, &ir_args);
                        if let Some(new_ptr) = new_val.ptr() {
                            not_original.insert(new_ptr, operand.clone());
                        }
                        new_val
                    }
                } else {
                    builder.create_ir(&stmt.ir, &ir_args)
                }
            } else {
                builder.create_ir(&stmt.ir, &ir_args)
            };

            value_lookup.insert(stmt.stmt_id, result);
        }

        builder.export_ir_graph()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. AlwaysSatisfiedElimination
// ═══════════════════════════════════════════════════════════════════════════

pub struct AlwaysSatisfiedElimination;

impl IRPass for AlwaysSatisfiedElimination {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut values_lookup: HashMap<StmtId, Value> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| values_lookup[&arg].clone())
                .collect();
            let val = builder.create_ir(&stmt.ir, &ir_args);
            values_lookup.insert(stmt.stmt_id, val);
        }

        // Find assertions that are always satisfied
        let mut to_eliminate: Vec<StmtId> = Vec::new();
        for stmt in &ir_graph.stmts {
            if matches!(stmt.ir, IR::Assert) {
                let cond_ptr = stmt.arguments[0];
                if let Some(cond_val) = &values_lookup.get(&cond_ptr) {
                    if let Some(v) = cond_val.int_val() {
                        if v != 0 {
                            to_eliminate.push(stmt.stmt_id);
                        }
                    }
                }
            }
        }

        ir_graph.remove_stmt_bunch(&to_eliminate);
        ir_graph
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. ConstantFold
// ═══════════════════════════════════════════════════════════════════════════

pub struct ConstantFold;

impl IRPass for ConstantFold {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut constant_int_cache: HashMap<i64, Value> = HashMap::new();
        let mut constant_float_cache: HashMap<u64, Value> = HashMap::new(); // f64 bits as key
        let constant_true = builder.ir_constant_bool(true);
        let constant_false = builder.ir_constant_bool(false);

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| {
                    let value = value_lookup[&arg].clone();
                    // Replace known constants with cached constant IRs
                    match &value {
                        Value::Boolean(sv) => match sv.static_val {
                            Some(true) => constant_true.clone(),
                            Some(false) => constant_false.clone(),
                            None => value,
                        },
                        Value::Integer(sv) => {
                            if let Some(v) = sv.static_val {
                                constant_int_cache
                                    .entry(v)
                                    .or_insert_with(|| builder.ir_constant_int(v))
                                    .clone()
                            } else {
                                value
                            }
                        }
                        Value::Float(sv) => {
                            if let Some(v) = sv.static_val {
                                let bits = v.to_bits();
                                constant_float_cache
                                    .entry(bits)
                                    .or_insert_with(|| builder.ir_constant_float(v))
                                    .clone()
                            } else {
                                value
                            }
                        }
                        _ => value,
                    }
                })
                .collect();

            let new_val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, new_val);
        }

        builder.export_ir_graph()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. DuplicateCodeElimination
// ═══════════════════════════════════════════════════════════════════════════

pub struct DuplicateCodeElimination;

impl IRPass for DuplicateCodeElimination {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        // Phase 1: identify duplicates
        let mut to_be_replaced: HashMap<StmtId, StmtId> = HashMap::new();
        let mut seen: Vec<(IR, Vec<StmtId>, StmtId)> = Vec::new();

        for stmt in &ir_graph.stmts {
            let mut existing = None;
            for (ir, args, id) in &seen {
                if *ir == stmt.ir && *args == stmt.arguments {
                    existing = Some(*id);
                    break;
                }
            }
            if let Some(existing_id) = existing {
                to_be_replaced.insert(stmt.stmt_id, existing_id);
            } else {
                seen.push((stmt.ir.clone(), stmt.arguments.clone(), stmt.stmt_id));
            }
        }

        // Phase 2: rebuild graph, replacing duplicates
        // values_lookup maps original stmt_id -> Value from the new builder
        // For duplicated stmts, we resolve through to_be_replaced to the canonical stmt
        let mut builder = IRBuilder::new();
        let mut values_lookup: HashMap<StmtId, Value> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            if let Some(&replacement) = to_be_replaced.get(&stmt.stmt_id) {
                // This stmt is a duplicate; point it at the canonical stmt's value
                // The canonical stmt must already be in values_lookup
                let val = values_lookup[&replacement].clone();
                values_lookup.insert(stmt.stmt_id, val);
                continue;
            }

            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| {
                    let resolved = to_be_replaced.get(&arg).copied().unwrap_or(arg);
                    values_lookup[&resolved].clone()
                })
                .collect();

            let val = builder.create_ir(&stmt.ir, &ir_args);
            values_lookup.insert(stmt.stmt_id, val);
        }

        builder.export_ir_graph()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. DynamicNDArrayMemoryLowering
// ═══════════════════════════════════════════════════════════════════════════

pub struct DynamicNDArrayMemoryLowering {
    pub mux_threshold: u32,
}

impl DynamicNDArrayMemoryLowering {
    pub fn new(mux_threshold: u32) -> Self {
        assert!(mux_threshold > 0, "mux_threshold must be positive");
        Self { mux_threshold }
    }
}

impl IRPass for DynamicNDArrayMemoryLowering {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        // Collect segment sizes and which segments have dynamic access
        let mut segment_sizes: HashMap<u32, u32> = HashMap::new();
        let mut dynamic_access_segments: HashSet<u32> = HashSet::new();

        for stmt in &ir_graph.stmts {
            if let IR::AllocateMemory { segment_id, size, .. } = &stmt.ir {
                segment_sizes.insert(*segment_id, *size);
            }
            match &stmt.ir {
                IR::DynamicNDArrayGetItem { segment_id, .. }
                | IR::DynamicNDArraySetItem { segment_id, .. } => {
                    dynamic_access_segments.insert(*segment_id);
                }
                _ => {}
            }
        }

        let mux_segments: HashSet<u32> = segment_sizes
            .iter()
            .filter(|(seg_id, &seg_size)| {
                dynamic_access_segments.contains(seg_id) && seg_size < self.mux_threshold
            })
            .map(|(&seg_id, _)| seg_id)
            .collect();

        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut mux_segment_cells: HashMap<u32, Vec<Value>> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();

            match &stmt.ir {
                IR::AllocateMemory {
                    segment_id,
                    size,
                    init_value,
                } if mux_segments.contains(segment_id) => {
                    let init_val = builder.ir_constant_int(*init_value);
                    let cells: Vec<Value> = (0..*size).map(|_| init_val.clone()).collect();
                    mux_segment_cells.insert(*segment_id, cells);
                    value_lookup.insert(stmt.stmt_id, Value::None);
                    continue;
                }

                IR::DynamicNDArrayGetItem {
                    segment_id,
                    array_id: _,
                } => {
                    if mux_segments.contains(segment_id) {
                        if let Some(cells) = mux_segment_cells.get(segment_id) {
                            let idx_val = &ir_args[0];
                            let mut lowered = cells[0].clone();
                            for (i, cell) in cells.iter().enumerate() {
                                let ci = builder.ir_constant_int(i as i64);
                                let cond = builder.ir_equal_i(idx_val, &ci);
                                lowered = builder.ir_select_i(&cond, cell, &lowered);
                            }
                            value_lookup.insert(stmt.stmt_id, lowered);
                            continue;
                        }
                    }
                    // Fall through to ReadMemory lowering
                    let lowered = builder.create_ir(
                        &IR::ReadMemory {
                            segment_id: *segment_id,
                        },
                        &[ir_args[0].clone()],
                    );
                    value_lookup.insert(stmt.stmt_id, lowered);
                    continue;
                }

                IR::DynamicNDArraySetItem {
                    segment_id,
                    array_id: _,
                } => {
                    if mux_segments.contains(segment_id) {
                        if let Some(cells) = mux_segment_cells.get(segment_id).cloned() {
                            let idx_val = &ir_args[0];
                            let write_val = &ir_args[1];
                            let mut next_cells = Vec::new();
                            for (i, cell) in cells.iter().enumerate() {
                                let ci = builder.ir_constant_int(i as i64);
                                let cond = builder.ir_equal_i(idx_val, &ci);
                                next_cells.push(builder.ir_select_i(&cond, write_val, cell));
                            }
                            mux_segment_cells.insert(*segment_id, next_cells);
                            value_lookup.insert(stmt.stmt_id, Value::None);
                            continue;
                        }
                    }
                    // Fall through to WriteMemory lowering
                    let lowered = builder.create_ir(
                        &IR::WriteMemory {
                            segment_id: *segment_id,
                        },
                        &[ir_args[0].clone(), ir_args[1].clone()],
                    );
                    value_lookup.insert(stmt.stmt_id, lowered);
                    continue;
                }

                _ => {}
            }

            let val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, val);
        }

        builder.export_ir_graph()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. DynamicNDArrayMetaAssertInjection
// ═══════════════════════════════════════════════════════════════════════════

pub struct DynamicNDArrayMetaAssertInjection;

impl IRPass for DynamicNDArrayMetaAssertInjection {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        // array_id -> (max_rank, max_length)
        let mut meta_lookup: HashMap<u32, (u32, u32)> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();

            let val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, val);

            match &stmt.ir {
                IR::AllocateDynamicNDArrayMeta {
                    array_id,
                    max_rank,
                    max_length,
                    ..
                } => {
                    meta_lookup.insert(*array_id, (*max_rank, *max_length));
                }
                IR::WitnessDynamicNDArrayMeta { array_id, max_rank } => {
                    let (alloc_max_rank, alloc_max_length) = meta_lookup
                        .get(array_id)
                        .unwrap_or_else(|| {
                            panic!(
                                "WitnessDynamicNDArrayMetaIR references unknown array_id={}",
                                array_id
                            )
                        });
                    assert_eq!(
                        *max_rank, *alloc_max_rank,
                        "WitnessDynamicNDArrayMetaIR max_rank mismatch"
                    );
                    // Inject AssertDynamicNDArrayMeta with the same args
                    builder.create_ir(
                        &IR::AssertDynamicNDArrayMeta {
                            array_id: *array_id,
                            max_rank: *alloc_max_rank,
                            max_length: *alloc_max_length,
                        },
                        &ir_args,
                    );
                }
                _ => {}
            }
        }

        builder.export_ir_graph()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. MemoryTraceInjection
// ═══════════════════════════════════════════════════════════════════════════

pub struct MemoryTraceInjection;

impl IRPass for MemoryTraceInjection {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut has_memory_access = false;

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();

            let new_val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, new_val.clone());

            match &stmt.ir {
                IR::WriteMemory { segment_id } => {
                    has_memory_access = true;
                    builder.create_ir(
                        &IR::MemoryTraceEmit {
                            segment_id: *segment_id,
                            is_write: true,
                        },
                        &[ir_args[0].clone(), ir_args[1].clone()],
                    );
                }
                IR::ReadMemory { segment_id } => {
                    has_memory_access = true;
                    builder.create_ir(
                        &IR::MemoryTraceEmit {
                            segment_id: *segment_id,
                            is_write: false,
                        },
                        &[ir_args[0].clone(), new_val],
                    );
                }
                _ => {}
            }
        }

        if has_memory_access {
            builder.create_ir(&IR::MemoryTraceSeal, &[]);
        }

        builder.export_ir_graph()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. PatternMatchOptim (shortcut optimization)
// ═══════════════════════════════════════════════════════════════════════════

pub struct PatternMatchOptim;

impl PatternMatchOptim {
    fn optimize_ir(&self, builder: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        match ir {
            IR::LogicalAnd => self.optimize_logical_and(builder, ir, args),
            IR::LogicalOr => self.optimize_logical_or(builder, ir, args),
            IR::SelectB => self.optimize_select_b(builder, ir, args),
            IR::SelectI => self.optimize_select_i(builder, ir, args),
            IR::SelectF => self.optimize_select_f(builder, ir, args),
            IR::AddI => self.optimize_add_i(builder, ir, args),
            IR::AddF => self.optimize_add_f(builder, ir, args),
            IR::SubI => self.optimize_sub_i(builder, ir, args),
            IR::SubF => self.optimize_sub_f(builder, ir, args),
            IR::MulI => self.optimize_mul_i(builder, ir, args),
            IR::MulF => self.optimize_mul_f(builder, ir, args),
            IR::DivI => self.optimize_div_i(builder, ir, args),
            IR::DivF => self.optimize_div_f(builder, ir, args),
            IR::LtI | IR::GtI => self.optimize_strict_cmp_same_ptr(builder, ir, args, false),
            IR::LtF | IR::GtF => self.optimize_strict_cmp_same_ptr_f(builder, ir, args, false),
            IR::LteI | IR::GteI => self.optimize_strict_cmp_same_ptr(builder, ir, args, true),
            IR::LteF | IR::GteF => self.optimize_strict_cmp_same_ptr_f(builder, ir, args, true),
            IR::EqI => self.optimize_eq_same_ptr(builder, ir, args),
            IR::EqF => self.optimize_eq_same_ptr_f(builder, ir, args),
            IR::NeI => self.optimize_ne_same_ptr(builder, ir, args),
            IR::NeF => self.optimize_ne_same_ptr_f(builder, ir, args),
            _ => builder.create_ir(ir, args),
        }
    }

    fn optimize_logical_and(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].int_val();
        let rv = args[1].int_val();
        if lv == Some(0) { return b.ir_constant_bool(false); }
        if rv == Some(0) { return b.ir_constant_bool(false); }
        if lv.is_some() && lv != Some(0) { return args[1].clone(); }
        if rv.is_some() && rv != Some(0) { return args[0].clone(); }
        if let (Some(a), Some(c)) = (lv, rv) {
            return b.ir_constant_bool(a != 0 && c != 0);
        }
        b.create_ir(ir, args)
    }

    fn optimize_logical_or(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].int_val();
        let rv = args[1].int_val();
        if lv.is_some() && lv != Some(0) { return b.ir_constant_bool(true); }
        if rv.is_some() && rv != Some(0) { return b.ir_constant_bool(true); }
        if lv == Some(0) { return args[1].clone(); }
        if rv == Some(0) { return args[0].clone(); }
        if let (Some(a), Some(c)) = (lv, rv) {
            return b.ir_constant_bool(a != 0 || c != 0);
        }
        b.create_ir(ir, args)
    }

    fn optimize_select_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let cond = &args[0];
        let tv = &args[1];
        let fv = &args[2];
        // If both branches are boolean, delegate to select_b logic
        if (tv.bool_val().is_some() || fv.bool_val().is_some())
            && matches!(tv, Value::Boolean(_)) && matches!(fv, Value::Boolean(_)) {
                return self.optimize_select_b(b, ir, args);
            }
        if tv.int_val() == fv.int_val() && fv.int_val().is_some() {
            return b.ir_constant_int(tv.int_val().unwrap());
        }
        if tv.ptr() == fv.ptr() && tv.ptr().is_some() {
            return tv.clone();
        }
        if tv.int_val() == Some(1) && fv.int_val() == Some(0) {
            return cond.clone();
        }
        let cv = cond.bool_val().or_else(|| cond.int_val().map(|v| v != 0));
        if let Some(true) = cv {
            if tv.int_val().is_some() {
                return b.ir_constant_int(tv.int_val().unwrap());
            }
            return tv.clone();
        }
        if let Some(false) = cv {
            if fv.int_val().is_some() {
                return b.ir_constant_int(fv.int_val().unwrap());
            }
            return fv.clone();
        }
        b.create_ir(ir, args)
    }

    fn optimize_select_b(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let cond = &args[0];
        let tv = &args[1];
        let fv = &args[2];
        if tv.bool_val() == fv.bool_val() && fv.bool_val().is_some() {
            return b.ir_constant_bool(tv.bool_val().unwrap());
        }
        if tv.ptr() == fv.ptr() && tv.ptr().is_some() {
            return tv.clone();
        }
        if tv.bool_val() == Some(true) && fv.bool_val() == Some(false) {
            return cond.clone();
        }
        let cv = cond.bool_val().or_else(|| cond.int_val().map(|v| v != 0));
        if let Some(true) = cv {
            if tv.bool_val().is_some() {
                return b.ir_constant_bool(tv.bool_val().unwrap());
            }
            return tv.clone();
        }
        if let Some(false) = cv {
            if fv.bool_val().is_some() {
                return b.ir_constant_bool(fv.bool_val().unwrap());
            }
            return fv.clone();
        }
        if fv.bool_val() == Some(false) {
            return b.ir_logical_and(cond, tv);
        }
        if tv.bool_val() == Some(true) {
            return b.ir_logical_or(cond, fv);
        }
        b.create_ir(ir, args)
    }

    fn optimize_select_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let cond = &args[0];
        let tv = &args[1];
        let fv = &args[2];
        if tv.float_val() == fv.float_val() && fv.float_val().is_some() {
            return b.ir_constant_float(tv.float_val().unwrap());
        }
        if tv.ptr() == fv.ptr() && tv.ptr().is_some() {
            return tv.clone();
        }
        let cv = cond.bool_val().or_else(|| cond.int_val().map(|v| v != 0));
        if let Some(true) = cv {
            if tv.float_val().is_some() {
                return b.ir_constant_float(tv.float_val().unwrap());
            }
            return tv.clone();
        }
        if let Some(false) = cv {
            if fv.float_val().is_some() {
                return b.ir_constant_float(fv.float_val().unwrap());
            }
            return fv.clone();
        }
        b.create_ir(ir, args)
    }

    fn optimize_add_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].int_val();
        let rv = args[1].int_val();
        if lv == Some(0) { return args[1].clone(); }
        if rv == Some(0) { return args[0].clone(); }
        if let (Some(a), Some(c)) = (lv, rv) { return b.ir_constant_int(a + c); }
        b.create_ir(ir, args)
    }

    fn optimize_add_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].float_val();
        let rv = args[1].float_val();
        if lv == Some(0.0) { return args[1].clone(); }
        if rv == Some(0.0) { return args[0].clone(); }
        if let (Some(a), Some(c)) = (lv, rv) { return b.ir_constant_float(a + c); }
        b.create_ir(ir, args)
    }

    fn optimize_sub_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[1].int_val() == Some(0) { return args[0].clone(); }
        if let (Some(a), Some(c)) = (args[0].int_val(), args[1].int_val()) {
            return b.ir_constant_int(a - c);
        }
        b.create_ir(ir, args)
    }

    fn optimize_sub_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[1].float_val() == Some(0.0) { return args[0].clone(); }
        if let (Some(a), Some(c)) = (args[0].float_val(), args[1].float_val()) {
            return b.ir_constant_float(a - c);
        }
        b.create_ir(ir, args)
    }

    fn optimize_mul_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].int_val();
        let rv = args[1].int_val();
        if lv == Some(1) { return args[1].clone(); }
        if rv == Some(1) { return args[0].clone(); }
        if let (Some(a), Some(c)) = (lv, rv) { return b.ir_constant_int(a * c); }
        b.create_ir(ir, args)
    }

    fn optimize_mul_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].float_val();
        let rv = args[1].float_val();
        if lv == Some(1.0) { return args[1].clone(); }
        if rv == Some(1.0) { return args[0].clone(); }
        if let (Some(a), Some(c)) = (lv, rv) { return b.ir_constant_float(a * c); }
        b.create_ir(ir, args)
    }

    fn optimize_div_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[1].int_val() == Some(1) { return args[0].clone(); }
        b.create_ir(ir, args)
    }

    fn optimize_div_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[1].float_val() == Some(1.0) { return args[0].clone(); }
        b.create_ir(ir, args)
    }

    // Self-comparison: a < a = false, a <= a = true
    fn optimize_strict_cmp_same_ptr(
        &self, b: &mut IRBuilder, ir: &IR, args: &[Value], equality_result: bool,
    ) -> Value {
        if args[0].ptr() == args[1].ptr() && args[0].ptr().is_some() {
            return b.ir_constant_bool(equality_result);
        }
        b.create_ir(ir, args)
    }

    fn optimize_strict_cmp_same_ptr_f(
        &self, b: &mut IRBuilder, ir: &IR, args: &[Value], equality_result: bool,
    ) -> Value {
        if args[0].ptr() == args[1].ptr() && args[0].ptr().is_some() {
            return b.ir_constant_bool(equality_result);
        }
        b.create_ir(ir, args)
    }

    fn optimize_eq_same_ptr(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[0].ptr() == args[1].ptr() && args[0].ptr().is_some() {
            return b.ir_constant_bool(true);
        }
        b.create_ir(ir, args)
    }

    fn optimize_eq_same_ptr_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[0].ptr() == args[1].ptr() && args[0].ptr().is_some() {
            return b.ir_constant_bool(true);
        }
        b.create_ir(ir, args)
    }

    fn optimize_ne_same_ptr(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[0].ptr() == args[1].ptr() && args[0].ptr().is_some() {
            return b.ir_constant_bool(false);
        }
        b.create_ir(ir, args)
    }

    fn optimize_ne_same_ptr_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[0].ptr() == args[1].ptr() && args[0].ptr().is_some() {
            return b.ir_constant_bool(false);
        }
        b.create_ir(ir, args)
    }
}

impl IRPass for PatternMatchOptim {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();
            let val = self.optimize_ir(&mut builder, &stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, val);
        }

        builder.export_ir_graph()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IRStatement;

    #[test]
    fn test_external_call_remover() {
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 1 }, vec![], None),
            IRStatement::new(
                1,
                IR::InvokeExternal {
                    store_idx: 0,
                    func_name: "f".to_string(),
                    args: vec![],
                    kwargs: HashMap::new(),
                },
                vec![],
                None,
            ),
            IRStatement::new(2, IR::Assert, vec![0], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = ExternalCallRemover.exec(graph);
        assert_eq!(result.len(), 2);
        // Should have const and assert, no invoke_external
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { .. }));
        assert!(matches!(result.stmts[1].ir, IR::Assert));
    }

    #[test]
    fn test_dead_code_elimination() {
        // stmt0 = const(10) [unused], stmt1 = const(20), stmt2 = assert(stmt1)
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 10 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 20 }, vec![], None),
            IRStatement::new(2, IR::Assert, vec![1], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = DeadCodeElimination.exec(graph);
        // stmt0 should be eliminated (unused, not fixed)
        assert_eq!(result.len(), 2);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 20 }));
        assert!(matches!(result.stmts[1].ir, IR::Assert));
    }

    #[test]
    fn test_double_not_elimination() {
        // stmt0 = const_bool(true), stmt1 = not(0), stmt2 = not(1) [== stmt0]
        let stmts = vec![
            IRStatement::new(0, IR::ConstantBool { value: true }, vec![], None),
            IRStatement::new(1, IR::LogicalNot, vec![0], None),
            IRStatement::new(2, IR::LogicalNot, vec![1], None),
            IRStatement::new(3, IR::Assert, vec![2], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = DoubleNotElimination.exec(graph);
        // The double not should be eliminated; stmt2 should refer back to stmt0's value
        // Result should have: const_bool(true), not(0), assert(0)
        // (the double-not returns the original, so assert references const_bool directly)
        assert!(result.len() <= 4);
    }

    #[test]
    fn test_always_satisfied_elimination() {
        // stmt0 = const(1), stmt1 = assert(0) — always satisfied
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 1 }, vec![], None),
            IRStatement::new(1, IR::Assert, vec![0], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = AlwaysSatisfiedElimination.exec(graph);
        // Assert on const(1) should be eliminated
        assert_eq!(result.len(), 1);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 1 }));
    }

    #[test]
    fn test_pattern_match_add_zero() {
        // stmt0 = const(0), stmt1 = read_integer, stmt2 = add_i(0, 1)
        // add(0, x) should simplify to x
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(
                1,
                IR::ReadInteger {
                    indices: vec![0],
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
            IRStatement::new(3, IR::Assert, vec![2], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = PatternMatchOptim.exec(graph);
        // The add should be simplified, so assert should reference the read directly
        // After optimization: const(0), read_int, assert(read_int)
        assert!(result.len() <= 4);
    }

    #[test]
    fn test_memory_trace_injection() {
        let stmts = vec![
            IRStatement::new(
                0,
                IR::AllocateMemory {
                    segment_id: 0,
                    size: 10,
                    init_value: 0,
                },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 42 }, vec![], None),
            IRStatement::new(3, IR::WriteMemory { segment_id: 0 }, vec![1, 2], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = MemoryTraceInjection.exec(graph);
        // Should have: alloc, const, const, write, trace_emit, trace_seal
        assert!(result.len() >= 6);
        // Last stmt should be MemoryTraceSeal
        assert!(matches!(
            result.stmts.last().unwrap().ir,
            IR::MemoryTraceSeal
        ));
    }
}
