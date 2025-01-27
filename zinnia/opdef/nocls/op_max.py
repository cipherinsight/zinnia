from typing import Any, Optional, List, Dict

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import OperatorCallError, TypeInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value


class MaxOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "max"

    @classmethod
    def get_name(cls) -> str:
        return "max"

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if len(kwargs) > 0:
            raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` does not accept keyword arguments")
        if len(args) == 0:
            raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` requires at least one argument")
        return {f"_n_{i}": arg for i, arg in enumerate(args)}

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        elements = [kwargs[f"_n_{i}"] for i in range(len(kwargs))]
        if len(elements) == 1:
            elements = builder.op_iter(elements[0]).values()
        if not all([e.type() == elements[0].type() for e in elements]):
            raise TypeInferenceError(dbg, f"All arguments for {self.get_name()} should have the same type")
        result = elements[0]
        for e in elements[1:]:
            result = builder.op_select(builder.op_bool_scalar(builder.op_less_than(result, e)), e, result)
        return result
