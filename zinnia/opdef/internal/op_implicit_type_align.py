from typing import List, Dict, Optional, Tuple

from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType, DTDescriptor, TupleDTDescriptor, ListDTDescriptor, \
    NDArrayDTDescriptor, BooleanType
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, NDArrayValue, ListValue, TupleValue, FloatValue


class ImplicitTypeAlignOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return f"implicit_type_align"

    @classmethod
    def get_name(cls) -> str:
        return "implicit_type_align"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs")
        ]

    @staticmethod
    def verify_align_ability(lhs: DTDescriptor, rhs: DTDescriptor) -> bool:
        if lhs == rhs:
            return True
        if lhs == IntegerType and rhs == FloatType:
            return True
        if lhs == FloatType and rhs == IntegerType:
            return True
        if lhs == BooleanType and rhs == IntegerType:
            return True
        if lhs == IntegerType and rhs == BooleanType:
            return True
        if lhs == BooleanType and rhs == FloatType:
            return True
        if lhs == FloatType and rhs == BooleanType:
            return True
        if isinstance(lhs, TupleDTDescriptor) and isinstance(rhs, TupleDTDescriptor):
            if len(lhs.elements_dtype) != len(rhs.elements_dtype):
                return False
            for l, r in zip(lhs.elements_dtype, rhs.elements_dtype):
                if not ImplicitTypeAlignOp.verify_align_ability(l, r):
                    return False
            return True
        if isinstance(lhs, ListDTDescriptor) and isinstance(rhs, ListDTDescriptor):
            if len(lhs.elements_dtype) != len(rhs.elements_dtype):
                return False
            for l, r in zip(lhs.elements_dtype, rhs.elements_dtype):
                if not ImplicitTypeAlignOp.verify_align_ability(l, r):
                    return False
            return True
        if (isinstance(lhs, ListDTDescriptor) or isinstance(lhs, TupleDTDescriptor)) and isinstance(rhs, NDArrayDTDescriptor):
            if len(lhs.elements_dtype) != rhs.shape[0]:
                return False
            if len(rhs.shape) > 1:
                sub_element_dest_type = NDArrayDTDescriptor(rhs.shape[1:], rhs.dtype)
                return all(
                    ImplicitTypeAlignOp.verify_align_ability(sub_element_type, sub_element_dest_type)
                    for sub_element_type in lhs.elements_dtype
                )
            return all(
                ImplicitTypeAlignOp.verify_align_ability(sub_element_type, rhs.dtype)
                for sub_element_type in lhs.elements_dtype
            )
        if isinstance(lhs, NDArrayDTDescriptor) and (isinstance(rhs, ListDTDescriptor) or isinstance(rhs, TupleDTDescriptor)):
            if len(rhs.elements_dtype) != lhs.shape[0]:
                return False
            if len(lhs.shape) > 1:
                sub_element_dest_type = NDArrayDTDescriptor(lhs.shape[1:], lhs.dtype)
                return all(
                    ImplicitTypeAlignOp.verify_align_ability(sub_element_type, sub_element_dest_type)
                    for sub_element_type in rhs.elements_dtype
                )
            return all(
                ImplicitTypeAlignOp.verify_align_ability(sub_element_type, lhs.dtype)
                for sub_element_type in rhs.elements_dtype
            )
        if isinstance(lhs, NDArrayDTDescriptor) and isinstance(rhs, NDArrayDTDescriptor):
            if lhs.shape != rhs.shape:
                return False
            return ImplicitTypeAlignOp.verify_align_ability(lhs.dtype, rhs.dtype)
        return False

    def recursive_build_implicit_type_align(self, builder: AbsIRBuilderInterface, lhs: Value, rhs: Value, dbg: Optional[DebugInfo] = None) -> Tuple[Value, Value]:
        if lhs.type() == rhs.type():
            return lhs, rhs
        elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
            return builder.op_float_cast(lhs), rhs
        elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
            return lhs, builder.op_float_cast(rhs)
        elif isinstance(lhs, TupleValue) and isinstance(rhs, TupleValue):
            assert len(lhs.types()) == len(rhs.types())
            new_values_l, new_values_r = [], []
            for l, r in zip(lhs.values(), rhs.values()):
                new_l, new_r = self.recursive_build_implicit_type_align(builder, l, r, dbg)
                new_values_l.append(new_l)
                new_values_r.append(new_r)
            return TupleValue(tuple(v.type() for v in new_values_l), tuple(new_values_l)), TupleValue(tuple(v.type() for v in new_values_r), tuple(new_values_r))
        elif isinstance(lhs, ListValue) and isinstance(rhs, ListValue):
            assert len(lhs.types()) == len(rhs.types())
            new_values_l, new_values_r = [], []
            for l, r in zip(lhs.values(), rhs.values()):
                new_l, new_r = self.recursive_build_implicit_type_align(builder, l, r, dbg)
                new_values_l.append(new_l)
                new_values_r.append(new_r)
            return ListValue(list(v.type() for v in new_values_l), list(new_values_l)), ListValue(list(v.type() for v in new_values_r), list(new_values_r))
        elif (isinstance(lhs, ListValue) or isinstance(lhs, TupleValue)) and isinstance(rhs, NDArrayValue):
            assert len(lhs.types()) == rhs.shape()[0]
            ndarray = builder.op_ndarray_asarray(lhs, dbg)
            if ndarray.dtype() == FloatType and rhs.dtype() == IntegerType:
                rhs = builder.op_ndarray_astype(rhs, builder.op_constant_class(FloatType), dbg)
            elif ndarray.dtype() == IntegerType and rhs.dtype() == FloatType:
                ndarray = builder.op_ndarray_astype(ndarray, builder.op_constant_class(FloatType), dbg)
            return ndarray, rhs
        elif isinstance(lhs, NDArrayValue) and (isinstance(rhs, ListValue) or isinstance(rhs, TupleValue)):
            assert len(rhs.types()) == lhs.shape()[0]
            ndarray = builder.op_ndarray_asarray(rhs, dbg)
            if ndarray.dtype() == FloatType and lhs.dtype() == IntegerType:
                lhs = builder.op_ndarray_astype(lhs, builder.op_constant_class(FloatType), dbg)
            elif ndarray.dtype() == IntegerType and lhs.dtype() == FloatType:
                ndarray = builder.op_ndarray_astype(ndarray, builder.op_constant_class(FloatType), dbg)
            return lhs, ndarray
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, NDArrayValue):
            if lhs.dtype() == FloatType and rhs.dtype() == IntegerType:
                return lhs, builder.op_ndarray_astype(rhs, builder.op_constant_class(FloatType), dbg)
            if lhs.dtype() == IntegerType and rhs.dtype() == FloatType:
                return builder.op_ndarray_astype(lhs, builder.op_constant_class(FloatType), dbg), rhs
            return lhs, rhs
        raise NotImplementedError()

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs = kwargs["lhs"]
        rhs = kwargs["rhs"]
        if not self.verify_align_ability(lhs.type(), rhs.type()):
            raise TypeInferenceError(dbg, f"Cannot implicit align the datatypes between {lhs.type()} and {rhs.type()}")
        new_lhs, new_rhs = self.recursive_build_implicit_type_align(builder, lhs, rhs, dbg)
        return TupleValue((new_lhs.type(), new_rhs.type()), (new_lhs, new_rhs))
