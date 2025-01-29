import copy
from typing import List, Dict, Optional

from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.debug.exception import TypeInferenceError
from zinnia.internal.internal_ndarray import InternalNDArray
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NDArrayValue, Value, ListValue, IntegerValue, FloatValue, TupleValue


class NDArray_AsarrayOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::asarray"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::asarray"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("val")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_val = kwargs["val"]
        if isinstance(the_val, NDArrayValue):
            return copy.deepcopy(the_val)
        if not isinstance(the_val, ListValue) and not isinstance(the_val, TupleValue):
            raise TypeInferenceError(dbg, f"Expected list or tuple here, got {the_val.type()}")
        def _recursive_collect_values(_val):
            if isinstance(_val, ListValue) or isinstance(_val, TupleValue):
                return [_recursive_collect_values(v) for v in _val.values()]
            return _val
        draft_list = _recursive_collect_values(_recursive_collect_values(the_val))
        if not InternalNDArray.is_nested_list_in_good_shape(draft_list):
            raise TypeInferenceError(dbg, f"To convert to NDArray, all sub-lists should be of the same shape.")
        the_shape = InternalNDArray.get_nested_list_shape(draft_list)
        internal_ndarray = InternalNDArray(the_shape, draft_list)
        inferred_dtype = IntegerType
        for v in internal_ndarray.flatten():
            if isinstance(v, IntegerValue):
                pass
            elif isinstance(v, FloatValue):
                inferred_dtype = FloatType
            else:
                raise TypeInferenceError(dbg, f"Expected int or float in the list, got {v.type()}")
        return NDArrayValue(the_shape, inferred_dtype, internal_ndarray)
