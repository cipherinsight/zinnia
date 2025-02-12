from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor, \
    TupleDTDescriptor, ListDTDescriptor, PoseidonHashedDTDescriptor, NoneDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, TupleValue, ListValue


class PlaceholderValueOp(AbstractOp):
    def __init__(self, dt: DTDescriptor):
        super().__init__()
        self.dt = dt

    def get_signature(self) -> str:
        return "placeholder_value"

    @classmethod
    def get_name(cls) -> str:
        return "placeholder_value"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        if isinstance(self.dt, NDArrayDTDescriptor):
            total_number_of_elements = 1
            for i in self.dt.shape:
                total_number_of_elements *= i
            dtype = self.dt.dtype
            values = []
            for i in range(total_number_of_elements):
                values.append(builder.op_placeholder_value(dtype, dbg))
            return NDArrayValue.from_shape_and_vector(self.dt.shape, dtype, values)
        elif isinstance(self.dt, TupleDTDescriptor):
            values = []
            for i, typ in enumerate(self.dt.elements_type):
                val = builder.op_placeholder_value(typ, dbg)
                values.append(val)
            return TupleValue(tuple(v.type() for v in values), tuple(values))
        elif isinstance(self.dt, ListDTDescriptor):
            values = []
            for i, typ in enumerate(self.dt.elements_type):
                val = builder.op_placeholder_value(typ, dbg)
                values.append(val)
            return ListValue(list(v.type() for v in values), list(values))
        elif isinstance(self.dt, IntegerDTDescriptor):
            return builder.ir_constant_int(0)
        elif isinstance(self.dt, FloatDTDescriptor):
            return builder.ir_constant_float(0.0)
        elif isinstance(self.dt, PoseidonHashedDTDescriptor):
            return builder.op_placeholder_value(self.dt.dtype, dbg)
        elif isinstance(self.dt, NoneDTDescriptor):
            return builder.op_constant_none()
        raise NotImplementedError()
