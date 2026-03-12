from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.optim_pass.memory_trace_injection import MemoryTraceInjectionIRPass
from zinnia.exec.exec_ctx import ExecutionContext
from zinnia.ir_def.defs.ir_allocate_memory import AllocateMemoryIR
from zinnia.ir_def.defs.ir_memory_trace_emit import MemoryTraceEmitIR
from zinnia.ir_def.defs.ir_memory_trace_seal import MemoryTraceSealIR
from zinnia.ir_def.defs.ir_read_memory import ReadMemoryIR
from zinnia.ir_def.defs.ir_write_memory import WriteMemoryIR


def test_memory_ir_roundtrip_serialization():
    ir_builder = IRBuilderImpl()
    addr = ir_builder.ir_constant_int(2)
    value = ir_builder.ir_constant_int(11)
    ir_builder.ir_allocate_memory(segment_id=1, size=16, init_value=7)
    ir_builder.ir_write_memory(segment_id=1, address=addr, value=value)
    ir_builder.ir_read_memory(segment_id=1, address=addr)
    ir_builder.ir_memory_trace_emit(segment_id=1, is_write=True, address=addr, value=value)
    ir_builder.ir_memory_trace_seal()

    stmts = ir_builder.export_ir_graph().export_stmts()
    restored = [IRStatement.import_from(stmt.export()) for stmt in stmts]

    assert any(isinstance(stmt.ir_instance, AllocateMemoryIR) for stmt in restored)
    assert any(isinstance(stmt.ir_instance, WriteMemoryIR) for stmt in restored)
    assert any(isinstance(stmt.ir_instance, ReadMemoryIR) for stmt in restored)
    assert any(isinstance(stmt.ir_instance, MemoryTraceEmitIR) for stmt in restored)
    assert any(isinstance(stmt.ir_instance, MemoryTraceSealIR) for stmt in restored)


def test_memory_trace_injection_pass_appends_trace_ir():
    ir_builder = IRBuilderImpl()
    addr = ir_builder.ir_constant_int(1)
    value = ir_builder.ir_constant_int(23)
    ir_builder.ir_allocate_memory(segment_id=0, size=8, init_value=0)
    ir_builder.ir_write_memory(segment_id=0, address=addr, value=value)
    ir_builder.ir_read_memory(segment_id=0, address=addr)

    graph = ir_builder.export_ir_graph()
    after_pass = MemoryTraceInjectionIRPass().exec(graph)
    stmts = after_pass.export_stmts()

    trace_emit_count = sum(1 for stmt in stmts if isinstance(stmt.ir_instance, MemoryTraceEmitIR))
    trace_seal_count = sum(1 for stmt in stmts if isinstance(stmt.ir_instance, MemoryTraceSealIR))

    assert trace_emit_count == 2
    assert trace_seal_count == 1


def test_execution_context_collects_memory_trace_rows():
    ir_builder = IRBuilderImpl()
    addr = ir_builder.ir_constant_int(3)
    value = ir_builder.ir_constant_int(99)
    ir_builder.ir_allocate_memory(segment_id=2, size=16, init_value=0)
    ir_builder.ir_write_memory(segment_id=2, address=addr, value=value)
    ir_builder.ir_read_memory(segment_id=2, address=addr)

    preprocess_graph = MemoryTraceInjectionIRPass().exec(ir_builder.export_ir_graph())
    preprocess_stmts = preprocess_graph.export_stmts()

    ctx = ExecutionContext(circuit_inputs=[], preprocess_stmts=preprocess_stmts, external_funcs={})
    _ = ctx.argparse()
    trace_rows = ctx.get_last_memory_trace()

    assert len(trace_rows) == 2
    assert trace_rows[0] == (2, 3, 0, 99, True)
    assert trace_rows[1] == (2, 3, 1, 99, False)
