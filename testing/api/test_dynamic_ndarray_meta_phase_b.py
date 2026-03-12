from zinnia.compile.backend.halo2_builder import Halo2ProgramBuilder
from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.optim_pass.dynamic_ndarray_meta_assert_injection import DynamicNDArrayMetaAssertInjectionIRPass
from zinnia.ir_def.defs.ir_assert_dynamic_ndarray_meta import AssertDynamicNDArrayMetaIR


def test_dynamic_ndarray_meta_assert_pass_injects_assert_ir():
    builder = IRBuilderImpl()
    rank = builder.ir_constant_int(2)
    offset = builder.ir_constant_int(0)
    shape_entries = [builder.ir_constant_int(4), builder.ir_constant_int(5)]
    stride_entries = [builder.ir_constant_int(5), builder.ir_constant_int(1)]

    builder.ir_allocate_dynamic_ndarray_meta(array_id=0, dtype_name="Integer", max_length=64, max_rank=2)
    builder.ir_witness_dynamic_ndarray_meta(
        array_id=0,
        max_rank=2,
        rank=rank,
        offset=offset,
        shape_entries=shape_entries,
        stride_entries=stride_entries,
    )

    graph = builder.export_ir_graph()
    lowered = DynamicNDArrayMetaAssertInjectionIRPass().exec(graph)
    stmts = lowered.export_stmts()

    assert any(isinstance(stmt.ir_instance, AssertDynamicNDArrayMetaIR) for stmt in stmts)


def test_halo2_codegen_contains_dynamic_ndarray_meta_constraints():
    builder = IRBuilderImpl()
    rank = builder.ir_constant_int(2)
    offset = builder.ir_constant_int(0)
    shape_entries = [builder.ir_constant_int(4), builder.ir_constant_int(5)]
    stride_entries = [builder.ir_constant_int(5), builder.ir_constant_int(1)]

    builder.ir_allocate_dynamic_ndarray_meta(array_id=1, dtype_name="Integer", max_length=128, max_rank=2)
    builder.ir_assert_dynamic_ndarray_meta(
        array_id=1,
        max_rank=2,
        max_length=128,
        rank=rank,
        offset=offset,
        shape_entries=shape_entries,
        stride_entries=stride_entries,
    )

    source = Halo2ProgramBuilder("dynamic_ndarray_meta_codegen", builder.export_ir_graph().export_stmts()).build()

    assert "rank_upper_" in source
    assert "shape_positive_" in source
    assert "stride_inactive_ok_" in source
