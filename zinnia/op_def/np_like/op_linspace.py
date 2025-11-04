from typing import Optional, List, Dict

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.type_sys import IntegerType, FloatType, NumberDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NumberValue, NDArrayValue, ListValue, TupleValue, IntegerValue, \
    ClassValue, NoneValue


class NP_LinspaceOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.linspace"

    @classmethod
    def get_name(cls) -> str:
        return "linspace"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("start"),
            AbstractOp._ParamEntry("stop"),
            AbstractOp._ParamEntry("num", default=True),
            AbstractOp._ParamEntry("endpoint", default=True),
            AbstractOp._ParamEntry("dtype", default=True),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    @staticmethod
    def _new_sample_op_lambda(
            builder: IRBuilderInterface,
            desired_type: NumberDTDescriptor,
            u: NumberValue, v: NumberValue,
            step: int, num: int,
            dbg: Optional[DebugInfo] = None
    ) -> NumberValue:
        if isinstance(u, IntegerValue):
            u = builder.ir_float_cast(u)
        if isinstance(v, IntegerValue):
            v = builder.ir_float_cast(v)
        _val = builder.ir_add_f(u, builder.ir_mul_f(builder.ir_sub_f(v, u), builder.ir_constant_float(step / num)))
        if desired_type == IntegerType:
            return builder.ir_int_cast(_val, dbg)
        return _val

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        start = kwargs["start"]
        stop = kwargs["stop"]
        num = kwargs.get("num", builder.ir_constant_int(50))
        endpoint = kwargs.get("endpoint", builder.ir_constant_int(1))
        dtype = kwargs.get("dtype", builder.op_constant_none())
        axis = kwargs.get("axis", builder.ir_constant_int(0))
        if isinstance(start, NumberValue):
            start = NDArrayValue.from_number(start)
        if isinstance(start, ListValue) or isinstance(start, TupleValue):
            start = builder.op_np_asarray(start, dbg)
        if isinstance(stop, NumberValue):
            stop = NDArrayValue.from_number(stop)
        if isinstance(stop, ListValue) or isinstance(stop, TupleValue):
            stop = builder.op_np_asarray(stop, dbg)
        if not isinstance(start, NDArrayValue):
            raise TypeInferenceError(dbg, f"Expected `start` to be a NDArray, got {start.type()}")
        if not isinstance(stop, NDArrayValue):
            raise TypeInferenceError(dbg, f"Expected `stop` to be a NDArray, got {stop.type()}")
        if not isinstance(num, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected `num` to be an Integer, got {num.type()}")
        if num.val(builder) is None:
            raise StaticInferenceError(dbg, f"The value to `num` cannot be statically inferred")
        if not isinstance(endpoint, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected `endpoint` to be a boolean, got {endpoint.type()}")
        if endpoint.val(builder) is None:
            raise StaticInferenceError(dbg, f"The value to `endpoint` cannot be statically inferred")
        inferred_dtype = FloatType  # the inferred dtype will never be integer
        if not isinstance(dtype, NoneValue) and not isinstance(dtype, ClassValue):
            raise TypeInferenceError(dbg, f"Expected `dtype` to be a type, got {endpoint.type()}")
        if isinstance(dtype, ClassValue):
            if dtype.val(builder) not in [IntegerType, FloatType]:
                raise TypeInferenceError(dbg, f"Expected `dtype` to be either `int` or `float`, got {dtype.val(builder)}")
            inferred_dtype = dtype.val(builder)
        if not isinstance(axis, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected `axis` to be an Integer, got {axis.type()}")
        if axis.val(builder) is None:
            raise StaticInferenceError(dbg, f"The value to `axis` cannot be statically inferred")
        start, stop = NDArrayValue.binary_broadcast(start, stop)
        samples = []
        for i in range(num.val(builder)):
            new_sample = NDArrayValue.binary(start, stop, inferred_dtype, lambda u, v: self._new_sample_op_lambda(
                builder, inferred_dtype, u, v, i, num.val(builder) - (1 if endpoint.val(builder) else 0), dbg
            ))
            samples.append(new_sample)
        return builder.op_np_stack(builder.op_square_brackets(samples, dbg), axis, dbg)
