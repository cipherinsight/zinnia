from typing import List, Dict, Optional, Tuple

from zinnia.compile.ast import ASTAnnotation
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor, \
    TupleDTDescriptor, ListDTDescriptor, PoseidonHashedDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue, TupleValue, ListValue


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

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        should_expose_public = False
        if self.kind == ASTAnnotation.Kind.PUBLIC:
            should_expose_public = True
        if isinstance(self.dt, NDArrayDTDescriptor):
            total_number_of_elements = 1
            for i in self.dt.shape:
                total_number_of_elements *= i
            dtype = self.dt.dtype
            values = []
            for i in range(total_number_of_elements):
                if isinstance(dtype, IntegerDTDescriptor):
                    _val = builder.ir_read_integer(self.indices + (i, ))
                    if should_expose_public:
                        builder.ir_expose_public_i(_val)
                    values.append(_val)
                elif isinstance(dtype, FloatDTDescriptor):
                    _val = builder.ir_read_float(self.indices + (i, ))
                    if should_expose_public:
                        builder.ir_expose_public_f(_val)
                    values.append(_val)
            return NDArrayValue.from_shape_and_vector(self.dt.shape, dtype, values)
        elif isinstance(self.dt, TupleDTDescriptor):
            values = []
            for i, typ in enumerate(self.dt.elements_dtype):
                val = builder.op_input(self.indices + (i, ), typ, '')
                values.append(val)
            return TupleValue(tuple(v.type() for v in values), tuple(values))
        elif isinstance(self.dt, ListDTDescriptor):
            values = []
            for i, typ in enumerate(self.dt.elements_dtype):
                val = builder.op_input(self.indices + (i, ), typ, '')
                values.append(val)
            return ListValue(list(v.type() for v in values), list(values))
        elif isinstance(self.dt, IntegerDTDescriptor):
            val = builder.ir_read_integer(self.indices)
            if should_expose_public:
                builder.ir_expose_public_i(val)
            return val
        elif isinstance(self.dt, FloatDTDescriptor):
            val = builder.ir_read_float(self.indices)
            if should_expose_public:
                builder.ir_expose_public_f(val)
            return val
        elif isinstance(self.dt, PoseidonHashedDTDescriptor):
            val = builder.op_input(self.indices + (0,), self.dt.dtype, '')
            provided_hash = builder.ir_read_hash(self.indices + (1,))
            builder.op_expose_public(provided_hash)
            builder.op_assert(builder.ir_equal_hash(builder.op_poseidon_hash(val), provided_hash), builder.op_constant_none())
            return val
        raise NotImplementedError()
