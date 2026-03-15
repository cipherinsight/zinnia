from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.optim_pass.dynamic_ndarray_memory_lowering import DynamicNDArrayMemoryLoweringIRPass
from zinnia.ir_def.defs.ir_dynamic_ndarray_get_item import DynamicNDArrayGetItemIR
from zinnia.ir_def.defs.ir_dynamic_ndarray_set_item import DynamicNDArraySetItemIR
from zinnia.ir_def.defs.ir_read_memory import ReadMemoryIR
from zinnia.ir_def.defs.ir_write_memory import WriteMemoryIR


def test_dynamic_ndarray_memory_lowering_rewrites_get_set_to_memory_ir():
    builder = IRBuilderImpl()
    addr = builder.ir_constant_int(4)
    val = builder.ir_constant_int(77)

    _ = builder.ir_dynamic_ndarray_get_item(array_id=1, segment_id=3, linear_address=addr)
    _ = builder.ir_dynamic_ndarray_set_item(array_id=1, segment_id=3, linear_address=addr, value=val)

    graph = builder.export_ir_graph()
    lowered = DynamicNDArrayMemoryLoweringIRPass().exec(graph)
    stmts = lowered.export_stmts()

    assert not any(isinstance(stmt.ir_instance, DynamicNDArrayGetItemIR) for stmt in stmts)
    assert not any(isinstance(stmt.ir_instance, DynamicNDArraySetItemIR) for stmt in stmts)
    assert any(isinstance(stmt.ir_instance, ReadMemoryIR) for stmt in stmts)
    assert any(isinstance(stmt.ir_instance, WriteMemoryIR) for stmt in stmts)

    read_stmt = next(stmt for stmt in stmts if isinstance(stmt.ir_instance, ReadMemoryIR))
    write_stmt = next(stmt for stmt in stmts if isinstance(stmt.ir_instance, WriteMemoryIR))
    assert read_stmt.ir_instance.segment_id == 3
    assert write_stmt.ir_instance.segment_id == 3
