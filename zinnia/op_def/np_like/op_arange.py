from typing import Dict, Any, Optional, List

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NoneValue, NumberValue, FloatValue, ClassValue, NDArrayValue


class NP_ARangeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.arange"

    @classmethod
    def get_name(cls) -> str:
        return "arange"

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(args) == 0:
            raise TypeInferenceError(dbg_i, "`arange` takes at least one argument")
        elif len(args) == 1:
            for kw in ["start", "stop"]:
                if kw in kwargs:
                    raise TypeInferenceError(dbg_i, f"Invalid argument {kw} here")
            parsed_kwargs = {"stop": args[0]}
        elif len(args) == 2:
            for kw in ["start", "stop"]:
                if kw in kwargs:
                    raise TypeInferenceError(dbg_i, f"Invalid argument {kw} here")
            parsed_kwargs = {"start": args[0], "stop": args[1]}
        elif len(args) == 3:
            for kw in ["start", "stop", "step"]:
                if kw in kwargs:
                    raise TypeInferenceError(dbg_i, f"Invalid argument {kw} here")
            parsed_kwargs = {"start": args[0], "stop": args[1], "step": args[2]}
        elif len(args) == 4:
            for kw in ["start", "stop", "step", "dtype"]:
                if kw in kwargs:
                    raise TypeInferenceError(dbg_i, f"Invalid argument {kw} here")
            parsed_kwargs = {"start": args[0], "stop": args[1], "step": args[2], "dtype": args[3]}
        else:
            raise TypeInferenceError(dbg_i, "`arange` takes at most 4 arguments")
        for k, v in kwargs.items():
            if k not in ["start", "stop", "step", "dtype"]:
                raise TypeInferenceError(dbg_i, f"Unexpected keyword argument {k}")
            parsed_kwargs[k] = v
        return parsed_kwargs

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        start = kwargs.get("start", builder.op_constant_none())
        stop = kwargs.get("stop", builder.op_constant_none())
        step = kwargs.get("step", builder.op_constant_none())
        dtype = kwargs.get("dtype", builder.op_constant_none())
        _start, _stop, _step = None, None, None
        if isinstance(start, NoneValue):
            _start = 0
        elif not isinstance(start, NumberValue):
            raise TypeInferenceError(dbg, "`arange` arguments must be of type `Integer` or `Float`")
        elif start.val(builder) is None:
            raise TypeInferenceError(dbg, "`arange` arguments must can be statically inferred")
        else:
            _start = start.val(builder)
        if not isinstance(stop, NumberValue):
            raise TypeInferenceError(dbg, "`arange` arguments must be of type `Integer` or `Float`")
        elif stop.val(builder) is None:
            raise TypeInferenceError(dbg, "`arange` arguments must can be statically inferred")
        else:
            _stop = stop.val(builder)
        if isinstance(step, NoneValue):
            _step = 1
        elif not isinstance(step, NumberValue):
            raise TypeInferenceError(dbg, "`arange` arguments must be of type `Integer` or `Float`")
        elif step.val(builder) is None:
            raise TypeInferenceError(dbg, "`arange` arguments must can be statically inferred")
        else:
            _step = step.val(builder)
        inferred_dtype = IntegerType
        if isinstance(start, FloatValue) or isinstance(stop, FloatValue) or isinstance(step, FloatValue):
            inferred_dtype = FloatType
        if not isinstance(dtype, NoneValue):
            if not isinstance(dtype, ClassValue):
                raise TypeInferenceError(dbg, "`dtype` must be a type")
            if dtype.val(builder) not in [IntegerType, FloatType]:
                raise TypeInferenceError(dbg, f"Unexpected dtype got: {dtype.val(builder)}")
            inferred_dtype = dtype.val(builder)
        elements = []
        current = _start
        while current < _stop:
            elements.append(current)
            current += _step
        if inferred_dtype == IntegerType:
            return NDArrayValue.from_shape_and_vector((len(elements),), inferred_dtype, [builder.ir_constant_int(int(i)) for i in elements])
        else:
            return NDArrayValue.from_shape_and_vector((len(elements),), inferred_dtype, [builder.ir_constant_float(i) for i in elements])
