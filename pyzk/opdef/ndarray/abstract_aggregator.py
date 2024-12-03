from typing import Dict, List, Any, Optional

from pyzk.exception.contextual import TypeInferenceError, StaticInferenceError
from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, NumberInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class AbstractAggregator(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("self"),
            _ParamEntry("axis", default=True),
        ]

    def aggregator_func(self, lhs: Any, rhs: Any) -> Any:
        raise NotImplementedError()

    def initial_func(self) -> Any:
        raise NotImplementedError()

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs["self"]
        the_axis = kwargs["axis"]
        if not isinstance(the_self, NDArrayInferenceDescriptor):
            raise TypeInferenceError(spi, "Param `self` must be of type `NDArray`")
        if the_axis is None:
            _axis = -1
        elif not isinstance(the_axis.type(), NumberDTDescriptor):
            raise TypeInferenceError(spi, "Param `axis` must be of type `Number`")
        elif the_axis.get() is None:
            raise StaticInferenceError(spi, "Cannot statically infer the value of param `axis`")
        else:
            _axis = the_axis.get()
        if _axis >= len(the_self.shape()):
            raise TypeInferenceError(spi, f"Invalid `axis` value for `{self.get_signature()}`. The axis number exceeds total number of dimensions of the ndarray")
        if _axis == -1:
            return NumberDTDescriptor()
        the_shape = the_self.shape()
        new_shape = tuple(x for i, x in enumerate(the_shape) if i != _axis)
        return NDArrayDTDescriptor(new_shape)

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs["self"]
        the_axis = kwargs["axis"]
        inferred_value = the_self.get().accumulate(
            the_axis.get() if the_axis is not None else -1,
            self.aggregator_func,
            self.initial_func
        )
        if not isinstance(inferred_value, NDArrayHelper):
            return NumberInferenceDescriptor(inferred_value)
        return NDArrayInferenceDescriptor(inferred_value.shape, inferred_value)
