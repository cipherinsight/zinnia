from typing import List, Dict, Optional, Tuple

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue, TupleValue, NoneValue, ListValue


class ExportExternalOp(AbstractOp):
    def __init__(self, for_which: int, key: int | str, indices: Tuple[int, ...]):
        super().__init__()
        self.for_which = for_which
        self.key = key
        self.indices = indices

    def get_signature(self) -> str:
        return f"export_external[{self.for_which}][{', '.join(map(str, self.indices))}]"

    @classmethod
    def get_name(cls) -> str:
        return "export_external"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def __eq__(self, other):
        return super().__eq__(other) and self.for_which == other.for_which and self.key == other.key and self.indices == other.indices

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_export_external_i(x, self.for_which, self.key, self.indices)
        elif isinstance(x, FloatValue):
            return builder.ir_export_external_f(x, self.for_which, self.key, self.indices)
        elif isinstance(x, TupleValue) or isinstance(x, ListValue):
            for i, val in enumerate(x.values()):
                builder.op_export_external(val, self.for_which, self.key, self.indices + (i, ))
            return NoneValue()
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            for i, v in enumerate(x.flattened_values()):
                builder.ir_export_external_i(v, self.for_which, self.key, self.indices + (i, ))
            return NoneValue()
        elif isinstance(x, NDArrayValue) and x.dtype() == FloatType:
            for i, v in enumerate(x.flattened_values()):
                builder.ir_export_external_f(v, self.for_which, self.key, self.indices + (i, ))
            return NoneValue()
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
