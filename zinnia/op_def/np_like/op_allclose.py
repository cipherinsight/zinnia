from typing import Dict, Optional, List

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import Value, NumberValue, NDArrayValue, ListValue, TupleValue
from zinnia.compile.type_sys import FloatType, IntegerType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class NP_AllCloseOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.allclose"

    @classmethod
    def get_name(cls) -> str:
        return "allclose"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
            AbstractOp._ParamEntry("rtol", True),
            AbstractOp._ParamEntry("atol", True),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        atol = kwargs.get("atol", builder.ir_constant_float(1e-08))
        rtol = kwargs.get("rtol", builder.ir_constant_float(1e-05))
        if isinstance(lhs, NumberValue):
            lhs = NDArrayValue.from_number(lhs)
        if isinstance(rhs, NumberValue):
            rhs = NDArrayValue.from_number(rhs)
        if isinstance(atol, NumberValue):
            atol = NDArrayValue.from_number(atol)
        if isinstance(rtol, NumberValue):
            rtol = NDArrayValue.from_number(rtol)
        if isinstance(lhs, ListValue) or isinstance(lhs, TupleValue):
            lhs = builder.op_np_asarray(lhs, dbg)
        if isinstance(rhs, ListValue) or isinstance(rhs, TupleValue):
            rhs = builder.op_np_asarray(rhs, dbg)
        if isinstance(atol, ListValue) or isinstance(atol, TupleValue):
            atol = builder.op_np_asarray(atol, dbg)
        if isinstance(rtol, ListValue) or isinstance(rtol, TupleValue):
            rtol = builder.op_np_asarray(rtol, dbg)
        if not isinstance(lhs, NDArrayValue):
            raise TypeInferenceError(dbg, f"Unsupported argument type for `lhs`: {lhs.type()}")
        if not isinstance(rhs, NDArrayValue):
            raise TypeInferenceError(dbg, f"Unsupported argument type for `rhs`: {rhs.type()}")
        if not isinstance(atol, NDArrayValue):
            raise TypeInferenceError(dbg, f"Unsupported argument type for `atol`: {atol.type()}")
        if not isinstance(rtol, NDArrayValue):
            raise TypeInferenceError(dbg, f"Unsupported argument type for `rtol`: {rtol.type()}")
        left = builder.op_abs(builder.op_subtract(lhs, rhs, dbg), dbg)
        right = builder.op_add(atol, builder.op_multiply(rtol, builder.op_abs(rhs, dbg), dbg), dbg)
        desired_dtype = FloatType if left.dtype() == FloatType else IntegerType
        if desired_dtype == IntegerType:
            right = builder.op_ndarray_astype(right, builder.op_constant_class(IntegerType), dbg)
        result = builder.op_less_than_or_equal(left, right, dbg)
        return builder.op_ndarray_all(result, dbg)
