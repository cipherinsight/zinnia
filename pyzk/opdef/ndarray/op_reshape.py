from typing import List, Dict, Optional

from pyzk.algo.ndarray_helper import NDArrayValueWrapper
from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.internal.dt_descriptor import IntegerType
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, NDArrayValue, TupleValue


class NDArray_ReshapeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::reshape"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::reshape"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("shape"),
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_shape = kwargs["shape"]
        assert isinstance(the_self, NDArrayValue)
        if not isinstance(the_shape, TupleValue):
            raise TypeInferenceError(dbg, f"`shape` of `{self.get_name()}` must be a Tuple")
        if not all(x == IntegerType for x in the_shape.types()):
            raise TypeInferenceError(dbg, f"`shape` of `{self.get_name()}` must be a Tuple of Integer")
        num_elements = 1
        for element in the_shape.values():
            if element is None:
                raise StaticInferenceError(dbg, f"Cannot statically infer the value of the argument `shape`")
            num_elements *= element
        num_elements_self = 1
        for element in the_self.shape():
            num_elements_self *= element
        if num_elements != num_elements_self:
            raise TypeInferenceError(dbg, f"Number of elements in `shape` must be equal to the number of elements in the original `NDArray`")
        flattened_values = the_self.get().flatten()
        new_shape = tuple(x.get() for x in the_shape.values())
        new_values = NDArrayValueWrapper.from_1d_values_and_shape(flattened_values, new_shape)
        return NDArrayValue(new_shape, the_self.dtype(), new_values)
