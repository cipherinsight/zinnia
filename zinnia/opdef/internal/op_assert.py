from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, NDArrayValue, NoneValue


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

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        operand = kwargs["test"]
        condition = kwargs.get("condition", builder.op_constant_none())
        if isinstance(condition, NoneValue):
            condition = builder.ir_constant_int(1)
        if not isinstance(condition, IntegerValue):
            raise TypeInferenceError(dbg, f"Internal Error: `condition` should be an integer value")
        if isinstance(operand, IntegerValue):
            if operand.val() == 0 and condition.val() != 0:
                raise StaticInferenceError(dbg, "Assertion is always unsatisfiable")
            return builder.ir_assert(builder.ir_select_i(condition, operand, builder.ir_constant_int(1)), dbg)
        elif isinstance(operand, NDArrayValue):
            test_val = builder.ir_constant_int(1)
            for val in operand.flattened_values():
                test_val = builder.ir_logical_and(test_val, builder.op_bool_cast(val, dbg))
            return builder.ir_assert(builder.ir_select_i(condition, test_val, builder.ir_constant_int(1)), dbg)
        raise TypeInferenceError(dbg, f"Type `{operand.type()}` is not supported on operator `{self.get_signature()}`. It only accepts an Integer value")
