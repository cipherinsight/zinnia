from itertools import product
from typing import List, Tuple

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import DynamicNDArrayValue, NumberValue, FloatValue, IntegerValue, BooleanValue


def logical_row_major_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return DynamicNDArrayValue._default_strides(shape)


def logical_decode_coords(linear: int, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> List[int]:
    return [((linear // stride) % dim) for dim, stride in zip(shape, strides)]


def logical_encode_coords(coords: List[int], strides: Tuple[int, ...]) -> int:
    out = 0
    for c, s in zip(coords, strides):
        out += c * s
    return out


def logical_num_elements(shape: Tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def iter_indices(shape: Tuple[int, ...]):
    if len(shape) == 0:
        yield ()
        return
    for idx in product(*[range(dim) for dim in shape]):
        yield idx


def default_like(builder: IRBuilderInterface, sample: NumberValue) -> NumberValue:
    if isinstance(sample, FloatValue):
        return builder.ir_constant_float(0.0)
    if isinstance(sample, BooleanValue):
        return builder.ir_constant_bool(False)
    if isinstance(sample, IntegerValue):
        return builder.ir_constant_int(0)
    return builder.ir_constant_int(0)


def flatten_logical_values(builder: IRBuilderInterface, arr: DynamicNDArrayValue) -> List[NumberValue]:
    shape = arr.logical_shape()
    strides = arr.logical_strides()
    offset = arr.logical_offset()
    base_values = arr.flattened_values()
    total = logical_num_elements(shape)
    if len(base_values) == 0:
        return []

    out = []
    row_strides = logical_row_major_strides(shape)
    for i in range(total):
        coords = logical_decode_coords(i, shape, row_strides)
        src_idx = offset + logical_encode_coords(coords, strides)
        if 0 <= src_idx < len(base_values):
            out.append(base_values[src_idx])
        else:
            out.append(default_like(builder, base_values[0]))
    return out