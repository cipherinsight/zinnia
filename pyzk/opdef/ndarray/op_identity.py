from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


class NDArray_IdentityOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::identity"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::identity"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("n")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        n = kwargs["n"]
        if not isinstance(n.type(), NumberDTDescriptor):
            raise TypeInferenceError(dbg_i, "Param `n` must be of type `Number`")
        if n.get() is None:
            raise StaticInferenceError(dbg_i, "Cannot statically infer the value of param `n`")
        if n.get() <= 0:
            raise TypeInferenceError(dbg_i, "Invalid `n` value, n must be greater than 0")
        return NDArrayDTDescriptor((n.get(), n.get()))

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        n = kwargs["n"].get()
        ndarray = NDArrayHelper.fill((n, n), lambda: 0)
        ndarray = ndarray.for_each(lambda pos, val: 1 if pos[0] == pos[1] else 0)
        return NDArrayInferenceDescriptor((n, n), ndarray)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        n = kwargs["n"]
        constant_0 = ir_builder.create_constant(0)
        constant_1 = ir_builder.create_constant(1)
        ndarray = NDArrayHelper.fill((n.val(), n.val()), lambda: constant_0)
        ndarray = ndarray.for_each(lambda pos, val: constant_1 if pos[0] == pos[1] else constant_0)
        return NDArrayFlattenDescriptor((n.val(), n.val()), ndarray)
