from typing import Dict, Any, Optional, List

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class RangeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "range"

    @classmethod
    def get_name(cls) -> str:
        return "range"

    def params_parse(self, spi: Optional[SourcePosInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(kwargs.items()) > 0:
            raise TypeInferenceError(spi, "`range` takes no keyword arguments")
        if len(args) == 0:
            raise TypeInferenceError(spi, "`range` takes at least one argument")
        if len(args) > 3:
            raise TypeInferenceError(spi, "`range` takes at most 3 arguments")
        if len(args) == 1:
            return {"start": None, "stop": args[0], "step": None}
        if len(args) == 2:
            return {"start": args[0], "stop": args[1], "step": None}
        if len(args) == 3:
            return {"start": args[0], "stop": args[1], "step": args[2]}
        raise NotImplementedError()

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        start, stop, step = kwargs["start"], kwargs["stop"], kwargs["step"]
        _start, _stop, _step = None, None, None
        if start is None:
            _start = 0
        elif not isinstance(start.type(), NumberDTDescriptor):
            raise TypeInferenceError(spi, "`range` param must be of type `Number`")
        elif start.get() is None:
            raise TypeInferenceError(spi, "`range` param must can be statically inferred")
        else:
            _start = start.get()
        if not isinstance(stop.type(), NumberDTDescriptor):
            raise TypeInferenceError(spi, "`range` param must be of type `Number`")
        elif stop.get() is None:
            raise TypeInferenceError(spi, "`range` param must can be statically inferred")
        else:
            _stop = stop.get()
        if step is None:
            _step = 1
        elif not isinstance(step.type(), NumberDTDescriptor):
            raise TypeInferenceError(spi, "`range` param must be of type `Number`")
        elif step.get() is None:
            raise TypeInferenceError(spi, "`range` param must can be statically inferred")
        else:
            _step = step.get()
        return NDArrayDTDescriptor((len(list(range(_start, _stop, _step))), ))

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        start, stop, step = kwargs["start"], kwargs["stop"], kwargs["step"]
        _start = 0 if start is None else start.get()
        _stop = stop.get()
        _step = 1 if step is None else step.get()
        result = list(range(_start, _stop, _step))
        return NDArrayInferenceDescriptor((len(result), ), NDArrayHelper((len(result), ), result))

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        start, stop, step = kwargs["start"], kwargs["stop"], kwargs["step"]
        _start = 0 if start is None else start.val()
        _stop = stop.val()
        _step = 1 if step is None else step.val()
        result = [ir_builder.create_constant(x) for x in range(_start, _stop, _step)]
        return NDArrayFlattenDescriptor((len(result), ), NDArrayHelper((len(result), ), result))
