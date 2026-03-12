from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.compile.type_sys import IntegerType
from zinnia.compile.type_sys.ndarray_bounds import infer_ndarray_compile_bounds_from_shape
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, TupleValue


class NDArray_ReshapeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.reshape"

    @classmethod
    def get_name(cls) -> str:
        return "reshape"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("shape"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_shape = kwargs["shape"]
        assert isinstance(the_self, NDArrayValue)
        if not isinstance(the_shape, TupleValue):
            raise TypeInferenceError(dbg, f"`shape` of `{self.get_name()}` must be a Tuple")
        if not all(x == IntegerType for x in the_shape.types()):
            raise TypeInferenceError(dbg, f"`shape` of `{self.get_name()}` must be a Tuple of Integer")
        bounds = infer_ndarray_compile_bounds_from_shape(the_shape, builder, dbg, self.get_name())
        num_elements = bounds.max_length
        num_elements_self = 1
        for element in the_self.shape():
            num_elements_self *= element
        if num_elements != num_elements_self:
            raise TypeInferenceError(dbg, f"Number of elements in `shape` must be equal to the number of elements in the original `NDArray`")
        flattened_values = the_self.get().flatten()
        new_shape = bounds.static_shape
        return NDArrayValue.from_shape_and_vector(new_shape, the_self.dtype(), flattened_values)
