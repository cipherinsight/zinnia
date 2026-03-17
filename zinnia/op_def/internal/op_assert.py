from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue, NDArrayValue, NoneValue


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

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        operand = kwargs["test"]
        condition = kwargs.get("condition", builder.op_constant_none())
        if isinstance(condition, NoneValue):
            condition = builder.ir_constant_bool(True)
        if not isinstance(condition, IntegerValue):
            raise TypeInferenceError(dbg, f"Internal Error: `condition` should be an integer value")
        if isinstance(operand, IntegerValue):
            asserted = builder.ir_select_i(condition, operand, builder.ir_constant_bool(True))
            solved = builder.smt_solve_constancy(asserted)
            if solved is False:
                raise StaticInferenceError(dbg, "Assertion is always unsatisfiable under current path condition")
            return builder.ir_assert(asserted, dbg)
        elif isinstance(operand, NDArrayValue):
            for val in operand.flattened_values():
                asserted = builder.ir_select_i(condition, builder.op_bool_cast(val), builder.ir_constant_bool(True))
                solved = builder.smt_solve_constancy(asserted)
                if solved is False:
                    raise StaticInferenceError(dbg, "Assertion is always unsatisfiable under current path condition")
                builder.ir_assert(asserted, dbg)
            return builder.op_constant_none()
        raise TypeInferenceError(dbg, f"Type `{operand.type()}` is not supported on operator `{self.get_signature()}`. It only accepts an Integer value")
