from typing import List, Dict, Optional

from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NoneValue


class PowOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "pow"

    @classmethod
    def get_name(cls) -> str:
        return "pow"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x"),
            AbstractOp._ParamEntry("exponent"),
            AbstractOp._ParamEntry("mod", default=True)
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        exponent = kwargs["exponent"]
        mod = kwargs.get("mod", builder.op_constant_none())
        power_result = builder.op_power(x, exponent, dbg)
        if not isinstance(mod, NoneValue):
            return builder.op_modulo(power_result, mod, dbg)
        return power_result
