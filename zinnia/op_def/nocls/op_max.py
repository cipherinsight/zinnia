from typing import Any, Optional, List, Dict

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import OperatorCallError, TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value


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

    def build(self, builder: IRBuilderInterface, op_args: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        elements = [val for key, val in op_args.get_kwargs().items() if key.startswith("_n_")]
        if len(elements) == 1:
            elements = builder.op_iter(elements[0]).values()
        if not all([e.type() == elements[0].type() for e in elements]):
            raise TypeInferenceError(dbg, f"All arguments for {self.get_name()} should have the same type")
        result = elements[0]
        for e in elements[1:]:
            result = builder.op_select(builder.op_bool_cast(builder.op_less_than(result, e)), e, result)
        return result
