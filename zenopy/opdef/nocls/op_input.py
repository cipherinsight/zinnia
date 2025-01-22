from typing import List, Dict, Optional, Tuple

from zenopy.compile.ast import ASTAnnotation
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor, \
    TupleDTDescriptor, ListDTDescriptor
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, NDArrayValue, HashedValue, TupleValue, ListValue


class InputOp(AbstractOp):
    def __init__(self, indices: Tuple[int, ...], dt: DTDescriptor, kind: str):
        super().__init__()
        self.indices = indices
        self.dt = dt
        self.kind = kind

    def get_signature(self) -> str:
        return "input"

    @classmethod
    def get_name(cls) -> str:
        return "input"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        should_expose_public = False
        should_return_hashed = False
        if self.kind == ASTAnnotation.Kind.PUBLIC:
            should_expose_public = True
        if self.kind == ASTAnnotation.Kind.HASHED:
            should_return_hashed = True
        if isinstance(self.dt, NDArrayDTDescriptor):
            the_idx = 0
            dtype = self.dt.dtype
            def _id_yield():
                nonlocal the_idx
                the_idx += 1
                if isinstance(dtype, IntegerDTDescriptor):
                    _val = reducer.ir_read_integer(self.indices + (the_idx - 1, ))
                    if should_expose_public:
                        reducer.ir_expose_public_i(_val)
                    return _val
                elif isinstance(dtype, FloatDTDescriptor):
                    _val = reducer.ir_read_float(self.indices + (the_idx - 1, ))
                    if should_expose_public:
                        reducer.ir_expose_public_f(_val)
                    return _val
                raise NotImplementedError()
            result = NDArrayValue.fill(self.dt.shape, dtype, _id_yield)
            if should_return_hashed:
                provided_hash = reducer.ir_read_hash(self.indices[0])
                reducer.op_expose_public(provided_hash)
                reducer.op_assert(reducer.ir_equal_i(reducer.op_hash(result), provided_hash), None)
                return HashedValue(result, provided_hash)
            return result
        elif isinstance(self.dt, TupleDTDescriptor):
            values = []
            for i, typ in enumerate(self.dt.elements_dtype):
                val = reducer.op_input(self.indices + (i, ), typ, '')
                values.append(val)
            return TupleValue(self.dt.elements_dtype, tuple(values))
        elif isinstance(self.dt, ListDTDescriptor):
            values = []
            for i, typ in enumerate(self.dt.elements_dtype):
                val = reducer.op_input(self.indices + (i, ), typ, '')
                values.append(val)
            return ListValue(self.dt.elements_dtype, list(values))
        elif isinstance(self.dt, IntegerDTDescriptor):
            val = reducer.ir_read_integer(self.indices)
            if should_expose_public:
                reducer.ir_expose_public_i(val)
            if should_return_hashed:
                provided_hash = reducer.ir_read_hash(self.indices[0])
                reducer.op_expose_public(provided_hash)
                reducer.op_assert(reducer.ir_equal_i(reducer.op_hash(val), provided_hash), None)
                return HashedValue(val, provided_hash)
            return val
        elif isinstance(self.dt, FloatDTDescriptor):
            val = reducer.ir_read_float(self.indices)
            if should_expose_public:
                reducer.ir_expose_public_f(val)
            if should_return_hashed:
                provided_hash = reducer.ir_read_hash(self.indices[0])
                reducer.op_expose_public(provided_hash)
                reducer.op_assert(reducer.ir_equal_i(reducer.op_hash(val), provided_hash), None)
                return HashedValue(val, provided_hash)
            return val
        raise NotImplementedError()
