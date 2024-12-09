from typing import Dict, List, Tuple, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, TupleInferenceDescriptor, \
    NDArrayInferenceDescriptor, NumberInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class AssignSliceOp(AbstractOp):
    def __init__(self, slicing_params_list: List[List[Tuple[int, ...]]]):
        super().__init__()
        self.slicing_params_list = slicing_params_list

    def get_signature(self) -> str:
        return "assign_slice"

    @classmethod
    def get_name(cls) -> str:
        return "assign_slice"

    def __eq__(self, other):
        def _compare_slicing_params_list(l1, l2) -> bool:
            if len(l1) != len(l2):
                return False
            for e1, e2 in zip(l1, l2):
                if len(e1) != len(e2):
                    return False
                for t1, t2 in zip(e1, e2):
                    if t1 != t2:
                        return False
            return True
        return super().__eq__(other) and _compare_slicing_params_list(self.slicing_params_list, other.slicing_params_list)

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        the_self = kwargs['self']
        the_value = kwargs['value']
        if isinstance(the_self, TupleInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"`{self.get_name()}` on `Tuple` is not allowed")
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
                raise TypeInferenceError(dbg_i, "In assign by slice, the value should be either `NDArray`, `Integer` or `Number`")
            check_result = the_self.get().check_slicing_assign(self.slicing_params_list, the_value.get())
            if check_result is not None:
                raise TypeInferenceError(dbg_i, f'Cannot assign by slice: {check_result}')
            if isinstance(the_value, NDArrayInferenceDescriptor) and the_self.dtype() != the_value.dtype():
                raise TypeInferenceError(dbg_i, f"Cannot assign by slice: the dtype of left NDArray ({the_self.dtype()}) is not equal to the dtype of right NDArray ({the_value.dtype()})")
            if isinstance(the_value, NumberInferenceDescriptor) and the_self.dtype() != the_value.dt:
                raise TypeInferenceError(dbg_i, f"Cannot assign by slice: the dtype of left NDArray ({the_self.dtype()}) is not equal to the datatype of the right ({the_value.dt})")
            sliced_result = the_self.get().slice_assign(self.slicing_params_list, the_value.get())
            return NDArrayDTDescriptor(sliced_result.shape, the_self.dtype())
        raise TypeInferenceError(dbg_i,f"Operator `{self.get_signature()}` can only be used on `Tuple` or `NDArray`")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        the_self = kwargs['self']
        the_value = kwargs['value']
        if isinstance(the_self, NDArrayInferenceDescriptor):
            sliced_result = the_self.get().slice_assign(self.slicing_params_list, the_value.get())
            return NDArrayInferenceDescriptor(sliced_result.shape, the_self.dtype(), sliced_result)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        the_self = kwargs['self']
        the_value = kwargs['value']
        if isinstance(the_self, NDArrayFlattenDescriptor):
            sliced_result = the_self.ptr().slice_assign(self.slicing_params_list, the_value.ptr())
            return NDArrayFlattenDescriptor(sliced_result.shape, the_self.dtype(), sliced_result)
        raise NotImplementedError()
