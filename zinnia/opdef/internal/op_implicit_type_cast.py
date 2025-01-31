from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType, DTDescriptor, TupleDTDescriptor, ListDTDescriptor, \
    NDArrayDTDescriptor, BooleanType
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, NDArrayValue, ListValue, TupleValue


class ImplicitTypeCastOp(AbstractOp):
    def __init__(self, dest: DTDescriptor):
        super().__init__()
        self.dest_type = dest

    def get_signature(self) -> str:
        return f"implicit_type_cast[{self.dest_type}]"

    @classmethod
    def get_name(cls) -> str:
        return "implicit_type_cast"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    @staticmethod
    def verify_cast_ability(src: DTDescriptor, dest: DTDescriptor) -> bool:
        if src == dest:
            return True
        if src == IntegerType and dest == FloatType:
            return True
        if src == BooleanType and dest == IntegerType:
            return True
        if src == BooleanType and dest == FloatType:
            return True
        if src == IntegerType and dest == BooleanType:
            return True
        if src == FloatType and dest == IntegerType:
            return False
        if isinstance(src, TupleDTDescriptor) and isinstance(dest, TupleDTDescriptor):
            if len(src.elements_dtype) != len(dest.elements_dtype):
                return False
            for s, d in zip(src.elements_dtype, dest.elements_dtype):
                if not ImplicitTypeCastOp.verify_cast_ability(s, d):
                    return False
            return True
        if isinstance(src, ListDTDescriptor) and isinstance(dest, ListDTDescriptor):
            if len(src.elements_dtype) != len(dest.elements_dtype):
                return False
            for s, d in zip(src.elements_dtype, dest.elements_dtype):
                if not ImplicitTypeCastOp.verify_cast_ability(s, d):
                    return False
            return True
        if (isinstance(src, ListDTDescriptor) or isinstance(src, TupleDTDescriptor)) and isinstance(dest, NDArrayDTDescriptor):
            if len(src.elements_dtype) != dest.shape[0]:
                return False
            if len(dest.shape) > 1:
                sub_element_dest_type = NDArrayDTDescriptor(dest.shape[1:], dest.dtype)
                return all(
                    ImplicitTypeCastOp.verify_cast_ability(sub_element_type, sub_element_dest_type)
                    for sub_element_type in src.elements_dtype
                )
            return all(
                ImplicitTypeCastOp.verify_cast_ability(sub_element_type, dest.dtype)
                for sub_element_type in src.elements_dtype
            )
        if isinstance(src, NDArrayDTDescriptor) and isinstance(dest, NDArrayDTDescriptor):
            if src.shape != dest.shape:
                return False
            return ImplicitTypeCastOp.verify_cast_ability(src.dtype, dest.dtype)
        return False

    def recursive_build_implicit_type_cast(self, builder: AbsIRBuilderInterface, src: Value, dest: DTDescriptor, dbg: Optional[DebugInfo] = None) -> Value:
        if src.type() == dest:
            return src
        if isinstance(src, IntegerValue) and dest == FloatType:
            return builder.op_float_cast(src)
        if isinstance(src, TupleValue) and isinstance(dest, TupleDTDescriptor):
            assert len(src.types()) == len(dest.elements_dtype)
            new_values = []
            for s, d in zip(src.values(), dest.elements_dtype):
                new_values.append(self.recursive_build_implicit_type_cast(builder, s, d, dbg))
            return TupleValue(tuple(v.type() for v in new_values), tuple(new_values))
        if isinstance(src, ListValue) and isinstance(dest, ListDTDescriptor):
            assert len(src.types()) == len(dest.elements_dtype)
            new_values = []
            for s, d in zip(src.values(), dest.elements_dtype):
                new_values.append(self.recursive_build_implicit_type_cast(builder, s, d, dbg))
            return TupleValue(tuple(v.type() for v in new_values), tuple(new_values))
        if (isinstance(src, ListValue) or isinstance(src, TupleValue)) and isinstance(dest, NDArrayDTDescriptor):
            assert len(src.types()) == dest.shape[0]
            if len(dest.shape) > 1:
                sub_element_dest_type = NDArrayDTDescriptor(dest.shape[1:], dest.dtype)
                new_values = []
                for s in src.values():
                    new_values.append(self.recursive_build_implicit_type_cast(builder, s, sub_element_dest_type, dbg))
                return NDArrayValue.stack(dest.dtype, 0, new_values)
            new_values = []
            for s in src.values():
                new_values.append(self.recursive_build_implicit_type_cast(builder, s, dest.dtype, dbg))
            return NDArrayValue.from_shape_and_vector((len(src.values()),), dest.dtype, new_values)
        if isinstance(src, NDArrayValue) and isinstance(dest, NDArrayDTDescriptor):
            if src.type() == dest:
                return src
            return builder.op_ndarray_astype(src, builder.op_constant_class(dest.dtype))
        raise NotImplementedError()

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if not self.verify_cast_ability(x.type(), self.dest_type):
            raise TypeInferenceError(dbg, f"Cannot implicit cast from {x.type()} to {self.dest_type}")
        return self.recursive_build_implicit_type_cast(builder, x, self.dest_type, dbg)
