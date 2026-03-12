from zinnia.compile.backend.halo2_builder import Halo2ProgramBuilder
from zinnia.compile.builder.builder_impl import IRBuilderImpl


def test_halo2_codegen_contains_memory_consistency_block():
    ir_builder = IRBuilderImpl()
    address = ir_builder.ir_constant_int(2)
    value = ir_builder.ir_constant_int(9)
    ir_builder.ir_allocate_memory(segment_id=0, size=8, init_value=0)
    ir_builder.ir_write_memory(segment_id=0, address=address, value=value)
    _ = ir_builder.ir_read_memory(segment_id=0, address=address)

    stmts = ir_builder.export_ir_graph().export_stmts()
    source = Halo2ProgramBuilder("memory_codegen_test", stmts).build()

    assert "mem_trace_unsorted" in source
    assert "mem_trace_sorted.sort_by_key" in source
    assert "Memory consistency constraints" in source
    assert "let mut mem_gp = ctx.load_constant(F::ONE);" in source
    assert "gate.assert_is_const(ctx, &mem_gp, &F::ONE);" in source


def test_halo2_codegen_rejects_trace_segment_without_allocation():
    ir_builder = IRBuilderImpl()
    address = ir_builder.ir_constant_int(1)
    value = ir_builder.ir_constant_int(3)
    ir_builder.ir_memory_trace_emit(segment_id=9, is_write=True, address=address, value=value)

    stmts = ir_builder.export_ir_graph().export_stmts()
    try:
        Halo2ProgramBuilder("memory_codegen_missing_segment", stmts).build()
        assert False, "Expected ValueError for unallocated segment"
    except ValueError as exc:
        assert "unallocated segment" in str(exc)
