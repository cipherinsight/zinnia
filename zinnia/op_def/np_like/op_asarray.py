import copy
from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.debug.exception import TypeInferenceError
from zinnia.internal.internal_ndarray import InternalNDArray
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NDArrayValue, Value, ListValue, IntegerValue, FloatValue, TupleValue, \
    ClassValue, NoneValue


class NP_AsarrayOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.asarray"

    @classmethod
    def get_name(cls) -> str:
        return "asarray"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("val"),
            AbstractOp._ParamEntry("dtype", default=True)
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_val = kwargs["val"]
        the_dtype = kwargs.get("dtype", builder.op_constant_none())
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
        if not isinstance(the_dtype, NoneValue):
            if not isinstance(the_dtype, ClassValue):
                raise TypeInferenceError(dbg, f"Expected dtype to be a type, got {the_dtype.type()}")
            if the_dtype.val() == IntegerType:
                internal_ndarray = internal_ndarray.unary(lambda u: builder.op_int_cast(u))
            elif the_dtype.val() == FloatType:
                internal_ndarray = internal_ndarray.unary(lambda u: builder.op_float_cast(u))
            else:
                raise TypeInferenceError(dbg, f"Expected dtype to be int or float, got {the_dtype.val()}")
            return NDArrayValue(the_shape, the_dtype.val(), internal_ndarray)
        inferred_dtype = IntegerType
        for v in internal_ndarray.flatten():
            if isinstance(v, IntegerValue):
                pass
            elif isinstance(v, FloatValue):
                inferred_dtype = FloatType
            else:
                raise TypeInferenceError(dbg, f"Expected int or float in the list, got {v.type()}")
        return NDArrayValue(the_shape, inferred_dtype, internal_ndarray)
