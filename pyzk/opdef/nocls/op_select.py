from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.internal.dt_descriptor import IntegerType, FloatType
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, IntegerValue, FloatValue, NDArrayValue, ListValue, TupleValue, NoneValue, \
    ClassValue


class SelectOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "select"

    @classmethod
    def get_name(cls) -> str:
        return "select"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("cond"),
            AbstractOp._ParamEntry("tv"),
            AbstractOp._ParamEntry("fv"),
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        cond = kwargs["cond"]
        tv, fv = kwargs["tv"], kwargs["fv"]
        if not isinstance(cond, IntegerValue):
            raise TypeInferenceError(dbg, f'In `{self.get_name()}`, argument `cond` must be an `Integer`')
        if tv.type() != fv.type():
            raise TypeInferenceError(dbg, f'In `{self.get_name()}`, arguments `tv` and `fv` must have the same type')
        if isinstance(tv, IntegerValue) and isinstance(fv, IntegerValue):
            return reducer.ir_select_i(cond, tv, fv)
        elif isinstance(tv, FloatValue) and isinstance(fv, FloatValue):
            return reducer.ir_select_f(cond, tv, fv)
        elif isinstance(tv, NDArrayValue) and isinstance(fv, NDArrayValue):
            if tv.dtype() == IntegerType:
                return NDArrayValue.binary(tv, fv, IntegerType, lambda x, y: reducer.ir_select_i(cond, x, y))
            elif tv.dtype() == FloatType:
                return NDArrayValue.binary(tv, fv, FloatType, lambda x, y: reducer.ir_select_f(cond, x, y))
        elif isinstance(tv, ListValue) and isinstance(fv, ListValue):
            values = [reducer.op_select(cond, tvv, fvv) for tvv, fvv in zip(tv.values(), fv.values())]
            return ListValue(tv.types(), values)
        elif isinstance(tv, TupleValue) and isinstance(fv, TupleValue):
            values = [reducer.op_select(cond, tvv, fvv) for tvv, fvv in zip(tv.values(), fv.values())]
            return TupleValue(tv.types(), tuple(values))
        elif isinstance(tv, NoneValue) and isinstance(fv, NoneValue):
            return NoneValue()
        elif isinstance(tv, ClassValue) and isinstance(fv, ClassValue):
            if cond.val() is None:
                raise StaticInferenceError(dbg, f'In `{self.get_name()}`, argument `cond` is not statically inferable, which is required for class selection')
            return tv if cond.val() else fv
        raise TypeInferenceError(dbg, f'In `{self.get_name()}`, unsupported types for `tv` and `fv`: {tv.type()} and {fv.type()}')
