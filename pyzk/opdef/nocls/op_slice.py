from typing import Dict, List, Tuple, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, NumberDTDescriptor, TupleDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, TupleFlattenDescriptor, NumberFlattenDescriptor, \
    NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor, NDArrayInferenceDescriptor, \
    NumberInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


class SliceOp(AbstractOp):
    def __init__(self, slicing_params: List[Tuple[int, ...]]):
        super().__init__()
        self.slicing_params = slicing_params

    def get_signature(self) -> str:
        return "slice" + "".join([f"[{param}]" for param in self.slicing_params])

    @classmethod
    def get_name(cls) -> str:
        return "slice"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs['self']
        if isinstance(the_self, TupleInferenceDescriptor):
            if len(self.slicing_params) != 1:
                raise TypeInferenceError(dbg_i, f"Only 1-dimensional slicing is supported by using `slice` on `Tuple`")
            if not 1 <= len(self.slicing_params[0]) <= 3:
                raise ValueError(f"Internal Error: Unexpected tuple length {len(self.slicing_params[0])}")
            if len(self.slicing_params[0]) == 1:
                if self.slicing_params[0][0] >= the_self.length():
                    raise TypeInferenceError(dbg_i, f"Tuple index {self.slicing_params[0][0]} out of range. The length of this tuple is {the_self.length()}")
                return NumberDTDescriptor()
            else:
                _start, _stop = self.slicing_params[0][0], self.slicing_params[0][1]
                _step = self.slicing_params[0][2] if len(self.slicing_params[0]) > 2 else 1
                return TupleDTDescriptor(len(range(the_self.length())[_start:_stop:_step]))
        elif isinstance(the_self, NDArrayInferenceDescriptor):
            if len(self.slicing_params) == 0:
                raise ValueError(f"Internal Error: Unexpected number of tuples {len(self.slicing_params)}")
            for param in self.slicing_params:
                if not 1 <= len(param) <= 3:
                    raise ValueError(f"Internal Error: Unexpected tuple length {len(param)}")
            check_result = the_self.get().check_slicing(self.slicing_params)
            if check_result is not None:
                raise TypeInferenceError(dbg_i, f'Cannot perform slicing: {check_result}')
            sliced_result = the_self.get().slice(self.slicing_params)
            if not isinstance(sliced_result, NDArrayHelper):
                return NumberDTDescriptor()
            return NDArrayDTDescriptor(sliced_result.shape)
        raise TypeInferenceError(dbg_i,"Operator `slice` can only be used on `Tuple` or `NDArray`")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs['self']
        if isinstance(the_self, TupleInferenceDescriptor):
            slicing = self.slicing_params[0]
            if len(slicing) == 1:
                return NumberInferenceDescriptor(the_self.get()[slicing[0]])
            elif len(slicing) == 2:
                the_result = the_self.get()[slicing[0]:slicing[1]]
                return TupleInferenceDescriptor(len(the_result), the_result)
            elif len(slicing) == 3:
                the_result = the_self.get()[slicing[0]:slicing[1]:slicing[2]]
                return TupleInferenceDescriptor(len(the_result), the_result)
        elif isinstance(the_self, NDArrayInferenceDescriptor):
            sliced_result = the_self.get().slice(self.slicing_params)
            if not isinstance(sliced_result, NDArrayHelper):
                return NumberInferenceDescriptor(sliced_result)
            return NDArrayInferenceDescriptor(sliced_result.shape, sliced_result)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs['self']
        if isinstance(the_self, TupleFlattenDescriptor):
            slicing = self.slicing_params[0]
            if len(slicing) == 1:
                return NumberFlattenDescriptor(the_self.ptr()[slicing[0]])
            elif len(slicing) == 2:
                the_result = the_self.ptr()[slicing[0]:slicing[1]]
                return TupleFlattenDescriptor(len(the_result), the_result)
            elif len(slicing) == 3:
                the_result = the_self.ptr()[slicing[0]:slicing[1]:slicing[2]]
                return TupleFlattenDescriptor(len(the_result), the_result)
        elif isinstance(the_self, NDArrayFlattenDescriptor):
            sliced_result = the_self.ptr().slice(self.slicing_params)
            if not isinstance(sliced_result, NDArrayHelper):
                return NumberFlattenDescriptor(sliced_result)
            return NDArrayFlattenDescriptor(sliced_result.shape, sliced_result)
        raise NotImplementedError()
