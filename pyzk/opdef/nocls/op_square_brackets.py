from typing import Dict, Any, Optional, List

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.abstract_op import AbstractOp
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class SquareBracketsOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "square_brackets"

    @classmethod
    def get_name(cls) -> str:
        return "square_brackets"

    def params_parse(self, spi: Optional[SourcePosInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(kwargs.items()) != 0:
            raise ValueError("Internal Error: `kwargs` Should be empty here")
        return {f'_{i}': arg for i, arg in enumerate(args)}

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        args = list(kwargs.values())
        if len(args) == 0:
            raise TypeInferenceError(spi, "Creating empty NDArray is not allowed, please provide at least one element in the square brackets")
        if all([isinstance(arg.type(), NumberDTDescriptor) for arg in args]):
            return NDArrayDTDescriptor((len(args), ))
        if not all([isinstance(arg.type(), NDArrayDTDescriptor) for arg in args]):
            raise TypeInferenceError(spi, "You can only create NDArray using square brackets on `Number` or `NDArray` types")
        for i, arg in enumerate(args[1:]):
            if arg.shape() != args[i - 1].shape():
                raise TypeInferenceError(spi,"Create NDArray using square brackets failed: elements shape mismatch")
        return NDArrayDTDescriptor((len(args), ) + args[0].shape(), )

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        args = list(kwargs.values())
        if all([isinstance(arg.type(), NumberDTDescriptor) for arg in args]):
            return NDArrayInferenceDescriptor((len(args), ), NDArrayHelper((len(args), ), [arg.get() for arg in args]))
        new_shape = (len(args), ) + args[0].shape()
        return NDArrayInferenceDescriptor(new_shape, NDArrayHelper.concat([arg.get() for arg in args]))

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        args = list(kwargs.values())
        if all([isinstance(arg.type(), NumberDTDescriptor) for arg in args]):
            return NDArrayFlattenDescriptor((len(args), ), NDArrayHelper((len(args), ), [arg.ptr() for arg in args]))
        new_shape = (len(args), ) + args[0].shape()
        return NDArrayFlattenDescriptor(new_shape, NDArrayHelper.concat([arg.ptr() for arg in args]))
