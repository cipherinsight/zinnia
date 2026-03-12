from typing import List, Optional, cast

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, ClassValue, NoneValue
from zinnia.compile.triplet.value.boolean import BooleanValue
from zinnia.compile.type_sys import FloatType, IntegerType, BooleanType, NumberDTDescriptor
from zinnia.compile.type_sys.ndarray_bounds import infer_ndarray_max_bounds_from_shape
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class DynamicNDArray_EyeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "DynamicNDArray.eye"

    @classmethod
    def get_name(cls) -> str:
        return "eye"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("n"),
            AbstractOp._ParamEntry("m"),
            AbstractOp._ParamEntry("dtype", True),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> DynamicNDArrayValue:
        n = kwargs["n"]
        m = kwargs["m"]
        dtype = kwargs.get("dtype", builder.op_constant_none())
        if not isinstance(n, IntegerValue):
            raise TypeInferenceError(dbg, "Param `n` must be of type `Number`")
        if not isinstance(m, IntegerValue):
            raise TypeInferenceError(dbg, "Param `m` must be of type `Number`")

        n_val = n.val(builder)
        m_val = m.val(builder)
        if n_val is not None and n_val <= 0:
            raise TypeInferenceError(dbg, "Invalid `n` value, n must be greater than 0")
        if m_val is not None and m_val <= 0:
            raise TypeInferenceError(dbg, "Invalid `m` value, m must be greater than 0")

        parsed_dtype = FloatType
        if not isinstance(dtype, NoneValue):
            if isinstance(dtype, ClassValue):
                parsed_dtype = dtype.val(builder)
            else:
                raise TypeInferenceError(dbg, "Invalid argument `dtype`, it must be a datatype")

        shape = builder.op_parenthesis([n, m])
        bounds = infer_ndarray_max_bounds_from_shape(shape, builder, dbg, self.get_name())
        max_n = infer_ndarray_max_bounds_from_shape(
            builder.op_parenthesis([n]),
            builder,
            dbg,
            self.get_name(),
        ).max_length
        max_m = infer_ndarray_max_bounds_from_shape(
            builder.op_parenthesis([m]),
            builder,
            dbg,
            self.get_name(),
        ).max_length
        max_writes = min(max_n, max_m)

        segment_id = len(getattr(builder, "stmts"))
        builder.ir_allocate_memory(segment_id=segment_id, size=bounds.max_length, init_value=0)

        placeholder_addr = builder.ir_constant_int(0)
        runtime_len = builder.op_multiply(n, m)
        for i in range(max_writes):
            i_val = builder.ir_constant_int(i)
            if i == 0:
                builder.ir_write_memory(
                    segment_id=segment_id,
                    address=placeholder_addr,
                    value=builder.ir_constant_int(1),
                )
                continue

            diag_addr = cast(IntegerValue, builder.op_add(builder.op_multiply(i_val, m), i_val))
            in_bounds = builder.op_less_than(diag_addr, runtime_len)
            row = builder.op_floor_divide(diag_addr, m)
            col = builder.op_modulo(diag_addr, m)
            is_diag = builder.op_equal(row, col)
            should_write = builder.op_logical_and(in_bounds, is_diag)
            should_write_b = cast(BooleanValue, builder.op_bool_cast(should_write))
            write_addr = builder.ir_select_i(should_write_b, diag_addr, placeholder_addr)
            placeholder_value = builder.ir_read_memory(segment_id=segment_id, address=placeholder_addr)
            write_val = builder.ir_select_i(should_write_b, builder.ir_constant_int(1), placeholder_value)
            builder.ir_write_memory(segment_id=segment_id, address=write_addr, value=write_val)

        values = []
        for i in range(bounds.max_length):
            idx = builder.ir_constant_int(i)
            read_val = builder.ir_read_memory(segment_id=segment_id, address=idx)
            if parsed_dtype == FloatType:
                values.append(builder.op_float_cast(read_val))
            elif parsed_dtype == IntegerType:
                values.append(read_val)
            elif parsed_dtype == BooleanType:
                values.append(builder.op_bool_cast(read_val))
            else:
                raise TypeInferenceError(dbg, f"Unsupported NDArray dtype {parsed_dtype}")

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            bounds.max_length,
            bounds.max_rank,
            cast(NumberDTDescriptor, parsed_dtype),
            values,
        )