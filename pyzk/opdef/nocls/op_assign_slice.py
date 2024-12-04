from typing import Dict, List, Tuple, Optional

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NDArrayDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor, NDArrayInferenceDescriptor, \
    NumberInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class AssignSliceOp(AbstractOp):
    def __init__(self, slicing_params_list: List[List[Tuple[int, ...]]]):
        super().__init__()
        self.slicing_params_list = slicing_params_list

    def get_signature(self) -> str:
        return "assign_slice"

    @classmethod
    def get_name(cls) -> str:
        return "assign_slice"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("self"),
            _ParamEntry("value")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs['self']
        the_value = kwargs['value']
        if isinstance(the_self, TupleInferenceDescriptor):
            raise TypeInferenceError(spi, f"`{self.get_name()}` on `Tuple` is not allowed")
        elif isinstance(the_self, NDArrayInferenceDescriptor):
            if len(self.slicing_params_list) == 0:
                raise ValueError(f"Internal Error: `slice` on `NDArray` should have the number of slicing params greater than 0")
            for slicing_params in self.slicing_params_list:
                if len(slicing_params) == 0:
                    raise ValueError(f"Internal Error: `slice` on `NDArray` should have the number of slicing params greater than 0")
                for slicing in slicing_params:
                    if not 1 <= len(slicing) <= 3:
                        raise ValueError(f'Internal Error: unexpected slicing found: {slicing}')
            if not isinstance(the_value, NDArrayInferenceDescriptor) and not isinstance(the_value, NumberInferenceDescriptor):
                raise TypeInferenceError(spi, "In assign by slice, the value should be either `NDArray` or `Number`")
            check_result = the_self.get().check_slicing_assign(self.slicing_params_list, the_value.get())
            if check_result is not None:
                raise TypeInferenceError(spi, f'Cannot assign by slice: {check_result}')
            sliced_result = the_self.get().slice_assign(self.slicing_params_list, the_value.get())
            return NDArrayDTDescriptor(sliced_result.shape)
        raise TypeInferenceError(spi,f"Operator `{self.get_signature()}` can only be used on `Tuple` or `NDArray`")

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs['self']
        the_value = kwargs['value']
        if isinstance(the_self, NDArrayInferenceDescriptor):
            sliced_result = the_self.get().slice_assign(self.slicing_params_list, the_value.get())
            return NDArrayInferenceDescriptor(sliced_result.shape, sliced_result)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs['self']
        the_value = kwargs['value']
        if isinstance(the_self, NDArrayFlattenDescriptor):
            sliced_result = the_self.ptr().slice_assign(self.slicing_params_list, the_value.ptr())
            return NDArrayFlattenDescriptor(sliced_result.shape, sliced_result)
        raise NotImplementedError()
