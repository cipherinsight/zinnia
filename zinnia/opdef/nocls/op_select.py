from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue, ListValue, TupleValue, NoneValue, \
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

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        cond = kwargs["cond"]
        _tv, _fv = kwargs["tv"], kwargs["fv"]
        if not isinstance(cond, IntegerValue):
            raise TypeInferenceError(dbg, f'In `{self.get_name()}`, argument `cond` must be an `Integer`')
        # TODO: implement implicit type casting in a better way
        tv, fv = _tv, _fv
        if isinstance(_tv, NDArrayValue) and isinstance(_fv, ListValue):
            tv, fv = _tv, builder.op_ndarray_asarray(_fv, dbg)
            fv = builder.op_ndarray_astype(fv, builder.op_constant_class(tv.dtype()), dbg)
        elif isinstance(_fv, NDArrayValue) and isinstance(_tv, ListValue):
            tv, fv = builder.op_ndarray_asarray(_tv, dbg), _fv
            tv = builder.op_ndarray_astype(tv, builder.op_constant_class(fv.dtype()), dbg)
        elif isinstance(_tv, NDArrayValue) and isinstance(_fv, NDArrayValue):
            if _tv.dtype() == IntegerType and _fv.dtype() == FloatType:
                tv, fv = builder.op_ndarray_astype(_tv, builder.op_constant_class(FloatType), dbg), _fv
            elif _tv.dtype() == FloatType and _fv.dtype() == IntegerType:
                tv, fv = _tv, builder.op_ndarray_astype(_fv, builder.op_constant_class(FloatType), dbg)
        elif isinstance(_tv, IntegerValue) and isinstance(_fv, FloatValue):
            tv, fv = builder.ir_float_cast(_tv, FloatType), _fv
        elif isinstance(_tv, FloatValue) and isinstance(_fv, IntegerValue):
            tv, fv = _tv, builder.ir_float_cast(_fv, FloatType)
        if tv.type() != fv.type():
            raise TypeInferenceError(dbg, f'In `{self.get_name()}`, arguments `tv` and `fv` must have the same type')
        if isinstance(tv, IntegerValue) and isinstance(fv, IntegerValue):
            return builder.ir_select_i(cond, tv, fv)
        elif isinstance(tv, FloatValue) and isinstance(fv, FloatValue):
            return builder.ir_select_f(cond, tv, fv)
        elif isinstance(tv, NDArrayValue) and isinstance(fv, NDArrayValue):
            if tv.dtype() == IntegerType:
                return NDArrayValue.binary(tv, fv, IntegerType, lambda x, y: builder.ir_select_i(cond, x, y))
            elif tv.dtype() == FloatType:
                return NDArrayValue.binary(tv, fv, FloatType, lambda x, y: builder.ir_select_f(cond, x, y))
        elif isinstance(tv, ListValue) and isinstance(fv, ListValue):
            values = [builder.op_select(cond, tvv, fvv) for tvv, fvv in zip(tv.values(), fv.values())]
            return ListValue(tv.types(), values)
        elif isinstance(tv, TupleValue) and isinstance(fv, TupleValue):
            values = [builder.op_select(cond, tvv, fvv) for tvv, fvv in zip(tv.values(), fv.values())]
            return TupleValue(tv.types(), tuple(values))
        elif isinstance(tv, NoneValue) and isinstance(fv, NoneValue):
            return NoneValue()
        elif isinstance(tv, ClassValue) and isinstance(fv, ClassValue):
            if cond.val() is None:
                raise StaticInferenceError(dbg, f'In `{self.get_name()}`, argument `cond` is not statically inferable, which is required for class selection')
            return tv if cond.val() else fv
        raise TypeInferenceError(dbg, f'In `{self.get_name()}`, unsupported types for `tv` and `fv`: {tv.type()} and {fv.type()}')
