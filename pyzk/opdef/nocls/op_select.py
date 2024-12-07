from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NumberDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NumberFlattenDescriptor, NDArrayFlattenDescriptor, \
    TupleFlattenDescriptor, NoneFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


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

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        cond = kwargs["cond"].type()
        tv, fv = kwargs["tv"].type(), kwargs["fv"].type()
        if not isinstance(cond, NumberDTDescriptor):
            raise TypeInferenceError(dbg_i, f'Param `cond` must be a number')
        if tv != fv:
            raise TypeInferenceError(dbg_i, f'The datatypes of `tv` and `fv` must be exactly the same')
        return tv

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        cond = kwargs["cond"]
        tv, fv = kwargs["tv"], kwargs["fv"]
        assert isinstance(cond, NumberInferenceDescriptor)
        if cond.get() is None:
            return tv.copy_reset()
        if cond.get() != 0:
            return tv
        return fv

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        cond = kwargs["cond"]
        tv, fv = kwargs["tv"], kwargs["fv"]
        assert isinstance(cond, NumberFlattenDescriptor)
        neg_cond = ir_builder.create_logical_not(cond.ptr())
        if isinstance(tv, NDArrayFlattenDescriptor):
            cond_mul_tv = tv.ptr().unary(lambda x: ir_builder.create_mul(cond.ptr(), x))
            cond_mul_fv = fv.ptr().unary(lambda x: ir_builder.create_mul(neg_cond, x))
            result = cond_mul_tv.binary(cond_mul_fv, lambda x, y: ir_builder.create_add(x, y))
            return NDArrayFlattenDescriptor(result.shape, result)
        elif isinstance(tv, TupleFlattenDescriptor):
            cond_mul_tv = tuple(ir_builder.create_mul(cond.ptr(), x) for x in tv.ptr())
            cond_mul_fv = tuple(ir_builder.create_mul(neg_cond, x) for x in fv.ptr())
            result = tuple(ir_builder.create_add(x, y) for x, y in zip(cond_mul_tv, cond_mul_fv))
            return TupleFlattenDescriptor(len(result), result)
        elif isinstance(tv, NoneFlattenDescriptor):
            return NoneFlattenDescriptor()
        elif isinstance(tv, NumberFlattenDescriptor):
            cond_mul_tv = ir_builder.create_mul(cond.ptr(), tv.ptr())
            cond_mul_fv = ir_builder.create_mul(neg_cond, fv.ptr())
            result = ir_builder.create_add(cond_mul_tv, cond_mul_fv)
            return NumberFlattenDescriptor(result)
        raise NotImplementedError()
