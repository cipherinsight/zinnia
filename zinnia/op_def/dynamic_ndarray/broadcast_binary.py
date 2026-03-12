from typing import Callable, List, Optional, cast

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, NumberValue
from zinnia.compile.type_sys import NumberDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.dynamic_ndarray.view_utils import default_like


def _as_ir_int(builder: IRBuilderInterface, x: IntegerValue) -> IntegerValue:
    v = x.val(builder)
    if v is not None:
        return builder.ir_constant_int(v)
    return x


def _runtime_rank(builder: IRBuilderInterface, arr: DynamicNDArrayValue) -> IntegerValue:
    return _as_ir_int(builder, arr.runtime_rank())


def _runtime_offset(builder: IRBuilderInterface, arr: DynamicNDArrayValue) -> IntegerValue:
    return _as_ir_int(builder, arr.runtime_offset())


def _runtime_shape(builder: IRBuilderInterface, arr: DynamicNDArrayValue) -> List[IntegerValue]:
    out: List[IntegerValue] = []
    for x in arr.runtime_shape_entries():
        out.append(_as_ir_int(builder, x))
    return out


def _runtime_stride(builder: IRBuilderInterface, arr: DynamicNDArrayValue) -> List[IntegerValue]:
    out: List[IntegerValue] = []
    for x in arr.runtime_stride_entries():
        out.append(_as_ir_int(builder, x))
    return out


def _normalize_meta_constraints(builder: IRBuilderInterface, arr: DynamicNDArrayValue, dbg: Optional[DebugInfo]) -> None:
    rank = _runtime_rank(builder, arr)
    max_rank = arr.max_rank()
    ge_zero = builder.op_greater_than_or_equal(rank, builder.ir_constant_int(0), dbg)
    le_max = builder.op_less_than_or_equal(rank, builder.ir_constant_int(max_rank), dbg)
    builder.op_assert(builder.op_logical_and(ge_zero, le_max, dbg), builder.op_constant_none(), dbg)

    offset = _runtime_offset(builder, arr)
    off_ge0 = builder.op_greater_than_or_equal(offset, builder.ir_constant_int(0), dbg)
    off_ltmax = builder.op_less_than(offset, builder.ir_constant_int(arr.max_length()), dbg)
    builder.op_assert(builder.op_logical_and(off_ge0, off_ltmax, dbg), builder.op_constant_none(), dbg)

    shape = _runtime_shape(builder, arr)
    stride = _runtime_stride(builder, arr)
    for i in range(arr.max_rank()):
        dim = shape[i]
        dim_ge1 = builder.op_greater_than_or_equal(dim, builder.ir_constant_int(1), dbg)
        dim_lemax = builder.op_less_than_or_equal(dim, builder.ir_constant_int(arr.max_length()), dbg)
        builder.op_assert(builder.op_logical_and(dim_ge1, dim_lemax, dbg), builder.op_constant_none(), dbg)
        stride_ge1 = builder.op_greater_than_or_equal(stride[i], builder.ir_constant_int(1), dbg)
        stride_lemax = builder.op_less_than_or_equal(stride[i], builder.ir_constant_int(arr.max_length()), dbg)
        builder.op_assert(builder.op_logical_and(stride_ge1, stride_lemax, dbg), builder.op_constant_none(), dbg)


def _write_src_memory(
    builder: IRBuilderInterface,
    arr: DynamicNDArrayValue,
    max_len: int,
) -> tuple[int, NumberValue]:
    values = arr.flattened_values()
    fallback = default_like(builder, cast(NumberValue, values[0])) if len(values) > 0 else builder.ir_constant_int(0)
    segment_id = len(getattr(builder, "stmts"))
    builder.ir_allocate_memory(segment_id=segment_id, size=max_len, init_value=0)
    for i in range(max_len):
        src_v = values[i] if i < len(values) else fallback
        builder.ir_write_memory(segment_id, builder.ir_constant_int(i), builder.op_int_cast(src_v))
    return segment_id, fallback


def _decode_coords(
    builder: IRBuilderInterface,
    linear: IntegerValue,
    shape: List[IntegerValue],
    rank: int,
    dbg: Optional[DebugInfo],
) -> List[IntegerValue]:
    coords: List[IntegerValue] = [builder.ir_constant_int(0) for _ in range(rank)]
    rem: IntegerValue = linear
    for k in range(rank - 1, -1, -1):
        dim = shape[k]
        coords[k] = cast(IntegerValue, builder.op_modulo(rem, dim, dbg))
        rem = cast(IntegerValue, builder.op_floor_divide(rem, dim, dbg))
    return coords


def _encode_addr(
    builder: IRBuilderInterface,
    coords: List[IntegerValue],
    stride: List[IntegerValue],
    offset: IntegerValue,
    rank: int,
    dbg: Optional[DebugInfo],
) -> IntegerValue:
    addr: IntegerValue = offset
    for k in range(rank):
        term = cast(IntegerValue, builder.op_multiply(coords[k], stride[k], dbg))
        addr = cast(IntegerValue, builder.op_add(addr, term, dbg))
    return addr


def dynamic_broadcast_binary(
    builder: IRBuilderInterface,
    lhs: DynamicNDArrayValue,
    rhs: DynamicNDArrayValue,
    out_dtype: NumberDTDescriptor,
    op_lambda: Callable[[NumberValue, NumberValue], NumberValue],
    dbg: Optional[DebugInfo] = None,
) -> DynamicNDArrayValue:
    _normalize_meta_constraints(builder, lhs, dbg)
    _normalize_meta_constraints(builder, rhs, dbg)

    out_max_rank = max(lhs.max_rank(), rhs.max_rank())
    lhs_shape = _runtime_shape(builder, lhs)
    rhs_shape = _runtime_shape(builder, rhs)
    lhs_stride = _runtime_stride(builder, lhs)
    rhs_stride = _runtime_stride(builder, rhs)

    # Left-pad shape/stride metadata to common max rank with broadcast-friendly defaults.
    lhs_pad = out_max_rank - lhs.max_rank()
    rhs_pad = out_max_rank - rhs.max_rank()
    lhs_shape_aligned: List[IntegerValue] = [builder.ir_constant_int(1) for _ in range(lhs_pad)] + lhs_shape
    rhs_shape_aligned: List[IntegerValue] = [builder.ir_constant_int(1) for _ in range(rhs_pad)] + rhs_shape
    lhs_stride_aligned: List[IntegerValue] = [builder.ir_constant_int(1) for _ in range(lhs_pad)] + lhs_stride
    rhs_stride_aligned: List[IntegerValue] = [builder.ir_constant_int(1) for _ in range(rhs_pad)] + rhs_stride

    out_shape: List[IntegerValue] = []
    for k in range(out_max_rank):
        ldim = lhs_shape_aligned[k]
        rdim = rhs_shape_aligned[k]
        l_eq_r = builder.op_equal(ldim, rdim, dbg)
        l_is_1 = builder.op_equal(ldim, builder.ir_constant_int(1), dbg)
        r_is_1 = builder.op_equal(rdim, builder.ir_constant_int(1), dbg)
        compat = builder.op_logical_or(builder.op_bool_cast(l_eq_r, dbg), builder.op_logical_or(builder.op_bool_cast(l_is_1, dbg), builder.op_bool_cast(r_is_1, dbg), dbg), dbg)
        builder.op_assert(compat, builder.op_constant_none(), dbg)
        out_dim = cast(IntegerValue, builder.op_select(builder.op_bool_cast(l_is_1, dbg), rdim, ldim, dbg))
        out_shape.append(out_dim)

    out_len: IntegerValue = builder.ir_constant_int(1)
    for dim in out_shape:
        out_len = cast(IntegerValue, builder.op_multiply(out_len, dim, dbg))

    out_max_len = lhs.max_length() * rhs.max_length()
    out_len_ok = builder.op_less_than_or_equal(out_len, builder.ir_constant_int(out_max_len), dbg)
    builder.op_assert(out_len_ok, builder.op_constant_none(), dbg)

    lhs_seg, lhs_fallback = _write_src_memory(builder, lhs, lhs.max_length())
    rhs_seg, rhs_fallback = _write_src_memory(builder, rhs, rhs.max_length())

    out_seg = len(getattr(builder, "stmts"))
    builder.ir_allocate_memory(segment_id=out_seg, size=out_max_len, init_value=0)

    out_values: List[NumberValue] = []
    lhs_offset = _runtime_offset(builder, lhs)
    rhs_offset = _runtime_offset(builder, rhs)

    for i in range(out_max_len):
        i_val = builder.ir_constant_int(i)
        active = builder.op_less_than(i_val, out_len, dbg)

        coords = _decode_coords(builder, i_val, out_shape, out_max_rank, dbg)

        lhs_coords: List[IntegerValue] = []
        rhs_coords: List[IntegerValue] = []
        for k in range(out_max_rank):
            ldim = lhs_shape_aligned[k]
            rdim = rhs_shape_aligned[k]
            l_is_1 = builder.op_equal(ldim, builder.ir_constant_int(1), dbg)
            r_is_1 = builder.op_equal(rdim, builder.ir_constant_int(1), dbg)
            lhs_coords.append(cast(IntegerValue, builder.op_select(builder.op_bool_cast(l_is_1, dbg), builder.ir_constant_int(0), coords[k], dbg)))
            rhs_coords.append(cast(IntegerValue, builder.op_select(builder.op_bool_cast(r_is_1, dbg), builder.ir_constant_int(0), coords[k], dbg)))

        lhs_addr = _encode_addr(builder, lhs_coords, lhs_stride_aligned, lhs_offset, out_max_rank, dbg)
        rhs_addr = _encode_addr(builder, rhs_coords, rhs_stride_aligned, rhs_offset, out_max_rank, dbg)

        lhs_addr_ok = builder.op_logical_and(
            builder.op_greater_than_or_equal(lhs_addr, builder.ir_constant_int(0), dbg),
            builder.op_less_than(lhs_addr, builder.ir_constant_int(lhs.max_length()), dbg),
            dbg,
        )
        rhs_addr_ok = builder.op_logical_and(
            builder.op_greater_than_or_equal(rhs_addr, builder.ir_constant_int(0), dbg),
            builder.op_less_than(rhs_addr, builder.ir_constant_int(rhs.max_length()), dbg),
            dbg,
        )
        builder.op_assert(builder.op_logical_or(builder.op_logical_not(builder.op_bool_cast(active, dbg), dbg), lhs_addr_ok, dbg), builder.op_constant_none(), dbg)
        builder.op_assert(builder.op_logical_or(builder.op_logical_not(builder.op_bool_cast(active, dbg), dbg), rhs_addr_ok, dbg), builder.op_constant_none(), dbg)

        lhs_read_addr = cast(IntegerValue, builder.op_select(builder.op_bool_cast(active, dbg), lhs_addr, builder.ir_constant_int(0), dbg))
        rhs_read_addr = cast(IntegerValue, builder.op_select(builder.op_bool_cast(active, dbg), rhs_addr, builder.ir_constant_int(0), dbg))

        lhs_iv = builder.ir_read_memory(lhs_seg, lhs_read_addr)
        rhs_iv = builder.ir_read_memory(rhs_seg, rhs_read_addr)

        lhs_num = cast(NumberValue, builder.op_implicit_type_cast(lhs_iv, lhs.dtype(), dbg))
        rhs_num = cast(NumberValue, builder.op_implicit_type_cast(rhs_iv, rhs.dtype(), dbg))
        out_num = cast(NumberValue, op_lambda(lhs_num, rhs_num))

        inactive_fill = default_like(builder, out_num)
        write_val = cast(NumberValue, builder.op_select(builder.op_bool_cast(active, dbg), out_num, inactive_fill, dbg))
        builder.ir_write_memory(out_seg, i_val, builder.op_int_cast(write_val))
        out_values.append(write_val)

    # Runtime row-major strides for output shape.
    out_stride: List[IntegerValue] = [builder.ir_constant_int(1) for _ in range(out_max_rank)]
    running = builder.ir_constant_int(1)
    for k in range(out_max_rank - 1, -1, -1):
        out_stride[k] = running
        running = cast(IntegerValue, builder.op_multiply(running, out_shape[k], dbg))

    return DynamicNDArrayValue.from_max_bounds_and_vector(
        max_length=out_max_len,
        max_rank=out_max_rank,
        dtype=out_dtype,
        values=out_values,
        logical_shape=(out_max_len,),
        logical_offset=0,
        logical_strides=(1,),
        runtime_logical_length=out_len,
        runtime_rank=builder.ir_constant_int(out_max_rank),
        runtime_shape_entries=out_shape,
        runtime_stride_entries=out_stride,
        runtime_offset=builder.ir_constant_int(0),
    )
