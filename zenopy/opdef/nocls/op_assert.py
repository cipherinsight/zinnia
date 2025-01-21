from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError, StaticInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue


class AssertOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "assert"

    @classmethod
    def get_name(cls) -> str:
        return "assert"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("test"),
            AbstractOp._ParamEntry("condition", True)
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        operand = kwargs["test"]
        condition = kwargs["condition"]
        if condition is None:
            condition = reducer.ir_constant_int(1)
        if not isinstance(condition, IntegerValue):
            raise TypeInferenceError(dbg, f"Internal Error: `condition` should be an integer value")
        if isinstance(operand, IntegerValue):
            if operand.val() == 0 and condition.val() != 0:
                raise StaticInferenceError(dbg, "Assertion is always unsatisfiable")
            return reducer.ir_assert(reducer.ir_select_i(condition, operand, reducer.ir_constant_int(1)), dbg)
        raise TypeInferenceError(dbg, f"Type `{operand.type()}` is not supported on operator `{self.get_signature()}`. It only accepts an Integer value")
