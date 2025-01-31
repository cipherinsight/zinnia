from typing import Dict, Any, Optional, List

from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import IntegerType
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, ListValue, IntegerValue


class RangeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "range"

    @classmethod
    def get_name(cls) -> str:
        return "range"

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(kwargs.items()) > 0:
            raise TypeInferenceError(dbg_i, "`range` takes no keyword arguments")
        if len(args) == 0:
            raise TypeInferenceError(dbg_i, "`range` takes at least one argument")
        if len(args) > 3:
            raise TypeInferenceError(dbg_i, "`range` takes at most 3 arguments")
        if len(args) == 1:
            return {"start": None, "stop": args[0], "step": None}
        if len(args) == 2:
            return {"start": args[0], "stop": args[1], "step": None}
        if len(args) == 3:
            return {"start": args[0], "stop": args[1], "step": args[2]}
        raise NotImplementedError()

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        start, stop, step = kwargs["start"], kwargs["stop"], kwargs["step"]
        _start, _stop, _step = None, None, None
        if start is None:
            _start = 0
        elif not isinstance(start, IntegerValue):
            raise TypeInferenceError(dbg, "`range` arguments must be of type `Integer`")
        elif start.val() is None:
            raise TypeInferenceError(dbg, "`range` arguments must can be statically inferred")
        else:
            _start = start.val()
        if not isinstance(stop, IntegerValue):
            raise TypeInferenceError(dbg, "`range` arguments must be of type `Integer`")
        elif stop.val() is None:
            raise TypeInferenceError(dbg, "`range` arguments must can be statically inferred")
        else:
            _stop = stop.val()
        if step is None:
            _step = 1
        elif not isinstance(step, IntegerValue):
            raise TypeInferenceError(dbg, "`range` arguments must be of type `Integer`")
        elif step.val() is None:
            raise TypeInferenceError(dbg, "`range` arguments must can be statically inferred")
        else:
            _step = step.val()
        values = list(range(_start, _stop, _step))
        values = [builder.ir_constant_int(v) for v in values]
        return ListValue([IntegerType for _ in values], values)
