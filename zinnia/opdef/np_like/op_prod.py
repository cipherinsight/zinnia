from typing import Dict, Optional, List

from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue, ListValue, TupleValue, NumberValue
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp


class NP_ProdOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.prod"

    @classmethod
    def get_name(cls) -> str:
        return "prod"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("a"),
            AbstractOp._ParamEntry("axis", True),
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        a, axis = kwargs["a"], kwargs.get("axis", builder.op_constant_none())
        if isinstance(a, NumberValue):
            a = NDArrayValue.from_number(a)
        if isinstance(a, ListValue) or isinstance(a, TupleValue):
            a = builder.op_ndarray_asarray(a, dbg)
        if not isinstance(a, NDArrayValue):
            raise TypeInferenceError(dbg, f"Operator `{self.get_name()}` on type `{a.type()}` is not defined. `a` must be a NDArray.")
        return builder.op_ndarray_prod(a, axis, dbg)
