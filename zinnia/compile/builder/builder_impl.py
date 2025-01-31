from typing import List, Dict, Optional, Tuple

from zinnia.compile.builder.ir_builder import IRBuilder
from zinnia.compile.builder.value import Value, ClassValue, TupleValue, ListValue, NoneValue, NDArrayValue, IntegerValue, \
    FloatValue, StringValue, NumberValue
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.type_sys import DTDescriptor
from zinnia.opdef.internal.op_implicit_type_align import ImplicitTypeAlignOp
from zinnia.opdef.internal.op_implicit_type_cast import ImplicitTypeCastOp
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.ir_op.ir_abs_f import AbsFIR
from zinnia.opdef.ir_op.ir_abs_i import AbsIIR
from zinnia.opdef.ir_op.ir_add_f import AddFIR
from zinnia.opdef.ir_op.ir_add_i import AddIIR
from zinnia.opdef.ir_op.ir_add_str import AddStrIR
from zinnia.opdef.ir_op.ir_assert import AssertIR
from zinnia.opdef.ir_op.ir_bool_cast import BoolCastIR
from zinnia.opdef.ir_op.ir_constant_float import ConstantFloatIR
from zinnia.opdef.ir_op.ir_constant_int import ConstantIntIR
from zinnia.opdef.ir_op.ir_constant_str import ConstantStrIR
from zinnia.opdef.ir_op.ir_cos_f import CosFIR
from zinnia.opdef.ir_op.ir_cosh_f import CosHFIR
from zinnia.opdef.ir_op.ir_div_f import DivFIR
from zinnia.opdef.ir_op.ir_div_i import DivIIR
from zinnia.opdef.ir_op.ir_eq_f import EqualFIR
from zinnia.opdef.ir_op.ir_eq_hash import EqualHashIR
from zinnia.opdef.ir_op.ir_eq_i import EqualIIR
from zinnia.opdef.ir_op.ir_exp_f import ExpFIR
from zinnia.opdef.ir_op.ir_export_external_f import ExportExternalFIR
from zinnia.opdef.ir_op.ir_export_external_i import ExportExternalIIR
from zinnia.opdef.ir_op.ir_expose_public_f import ExposePublicFIR
from zinnia.opdef.ir_op.ir_expose_public_i import ExposePublicIIR
from zinnia.opdef.ir_op.ir_float_cast import FloatCastIR
from zinnia.opdef.ir_op.ir_floor_divide_f import FloorDivFIR
from zinnia.opdef.ir_op.ir_floor_divide_i import FloorDivIIR
from zinnia.opdef.ir_op.ir_gt_f import GreaterThanFIR
from zinnia.opdef.ir_op.ir_gt_i import GreaterThanIIR
from zinnia.opdef.ir_op.ir_gte_f import GreaterThanOrEqualFIR
from zinnia.opdef.ir_op.ir_gte_i import GreaterThanOrEqualIIR
from zinnia.opdef.ir_op.ir_poseidon_hash import PoseidonHashIR
from zinnia.opdef.ir_op.ir_int_cast import IntCastIR
from zinnia.opdef.ir_op.ir_invoke_external import InvokeExternalIR
from zinnia.opdef.ir_op.ir_log_f import LogFIR
from zinnia.opdef.ir_op.ir_logical_and import LogicalAndIR
from zinnia.opdef.ir_op.ir_logical_not import LogicalNotIR
from zinnia.opdef.ir_op.ir_logical_or import LogicalOrIR
from zinnia.opdef.ir_op.ir_lt_f import LessThanFIR
from zinnia.opdef.ir_op.ir_lt_i import LessThanIIR
from zinnia.opdef.ir_op.ir_lte_f import LessThanOrEqualFIR
from zinnia.opdef.ir_op.ir_lte_i import LessThanOrEqualIIR
from zinnia.opdef.ir_op.ir_mod_f import ModFIR
from zinnia.opdef.ir_op.ir_mod_i import ModIIR
from zinnia.opdef.ir_op.ir_mul_f import MulFIR
from zinnia.opdef.ir_op.ir_mul_i import MulIIR
from zinnia.opdef.ir_op.ir_ne_f import NotEqualFIR
from zinnia.opdef.ir_op.ir_ne_i import NotEqualIIR
from zinnia.opdef.ir_op.ir_pow_f import PowFIR
from zinnia.opdef.ir_op.ir_pow_i import PowIIR
from zinnia.opdef.ir_op.ir_print import PrintIR
from zinnia.opdef.ir_op.ir_read_float import ReadFloatIR
from zinnia.opdef.ir_op.ir_read_hash import ReadHashIR
from zinnia.opdef.ir_op.ir_read_integer import ReadIntegerIR
from zinnia.opdef.ir_op.ir_select_f import SelectFIR
from zinnia.opdef.ir_op.ir_select_i import SelectIIR
from zinnia.opdef.ir_op.ir_sign_f import SignFIR
from zinnia.opdef.ir_op.ir_sign_i import SignIIR
from zinnia.opdef.ir_op.ir_sin_f import SinFIR
from zinnia.opdef.ir_op.ir_sinh_f import SinHFIR
from zinnia.opdef.ir_op.ir_sqrt_f import SqrtFIR
from zinnia.opdef.ir_op.ir_str_f import StrFIR
from zinnia.opdef.ir_op.ir_str_i import StrIIR
from zinnia.opdef.ir_op.ir_sub_f import SubFIR
from zinnia.opdef.ir_op.ir_sub_i import SubIIR
from zinnia.opdef.ir_op.ir_tan_f import TanFIR
from zinnia.opdef.ir_op.ir_tanh_f import TanHFIR
from zinnia.opdef.np_like.op_asarray import NP_AsarrayOp
from zinnia.opdef.ndarray.op_astype import NDArray_AsTypeOp
from zinnia.opdef.ndarray.op_get_item import NDArray_GetItemOp
from zinnia.opdef.ndarray.op_set_item import NDArray_SetItemOp
from zinnia.opdef.ndarray.op_tolist import NDArray_ToListOp
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.opdef.nocls.op_abs import AbsOp
from zinnia.opdef.arithmetic.op_add import AddOp
from zinnia.opdef.internal.op_assert import AssertOp
from zinnia.opdef.nocls.op_bool_cast import BoolCastOp
from zinnia.opdef.arithmetic.op_div import DivOp
from zinnia.opdef.arithmetic.op_eq import EqualOp
from zinnia.opdef.math.op_exp import Math_ExpOp
from zinnia.opdef.internal.op_export_external import ExportExternalOp
from zinnia.opdef.internal.op_expose_public import ExposePublicOp
from zinnia.opdef.nocls.op_float_cast import FloatCastOp
from zinnia.opdef.arithmetic.op_floor_divide import FloorDivideOp
from zinnia.opdef.arithmetic.op_gt import GreaterThanOp
from zinnia.opdef.arithmetic.op_gte import GreaterThanOrEqualOp
from zinnia.opdef.internal.op_poseidon_hash import PoseidonHashOp
from zinnia.opdef.internal.op_input import InputOp
from zinnia.opdef.nocls.op_int_cast import IntCastOp
from zinnia.opdef.internal.op_iter import IterOp
from zinnia.opdef.nocls.op_list import ListOp
from zinnia.opdef.math.op_log import Math_LogOp
from zinnia.opdef.arithmetic.op_lt import LessThanOp
from zinnia.opdef.arithmetic.op_lte import LessThanOrEqualOp
from zinnia.opdef.arithmetic.op_mat_mul import MatMulOp
from zinnia.opdef.nocls.op_max import MaxOp
from zinnia.opdef.nocls.op_min import MinOp
from zinnia.opdef.arithmetic.op_mod import ModOp
from zinnia.opdef.arithmetic.op_mul import MulOp
from zinnia.opdef.arithmetic.op_ne import NotEqualOp
from zinnia.opdef.np_like.op_logical_not import NP_LogicalNotOp
from zinnia.opdef.nocls.op_pow import PowOp
from zinnia.opdef.internal.op_select import SelectOp
from zinnia.opdef.internal.op_get_item import GetItemOp
from zinnia.opdef.internal.op_set_item import SetItemOp
from zinnia.opdef.internal.op_sign import Math_SignOp
from zinnia.opdef.math.op_sqrt import Math_SqrtOp
from zinnia.opdef.nocls.op_str import StrOp
from zinnia.opdef.arithmetic.op_sub import SubOp
from zinnia.opdef.nocls.op_tuple import TupleOp
from zinnia.opdef.arithmetic.op_uadd import UAddOp
from zinnia.opdef.arithmetic.op_usub import USubOp


class IRBuilderImpl(IRBuilder):
    def __init__(self) -> None:
        super().__init__()

    def create_op(self, operator: AbstractOp, args: List[Value], kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        kwargs = operator.argparse(dbg, args, kwargs)
        return operator.build(self, kwargs, dbg)

    def create_ir(self, operator: AbstractIR, args: List[Value], kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        val, stmt = operator.build_ir(len(self.stmts), operator.argparse(dbg, args, kwargs), dbg)
        self.stmts.append(stmt)
        return val

    def op_select(self, condition: Value, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = SelectOp()
        kwargs = op.argparse(dbg, [condition, a, b], {})
        return op.build(self, kwargs, dbg)

    def op_assert(self, test: Value, condition: IntegerValue | None, dbg: Optional[DebugInfo] = None) -> Value:
        op = AssertOp()
        kwargs = op.argparse(dbg, [test] + ([condition] if condition is not None else []), {})
        return op.build(self, kwargs, dbg)

    def op_less_than(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = LessThanOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_less_than_or_equal(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = LessThanOrEqualOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_equal(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = EqualOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_not_equal(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = NotEqualOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_greater_than(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = GreaterThanOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_greater_than_or_equal(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = GreaterThanOrEqualOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_add(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = AddOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_subtract(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = SubOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_multiply(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = MulOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_divide(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = DivOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_floor_divide(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = FloorDivideOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_mat_mul(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = MatMulOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_modulo(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = ModOp()
        kwargs = op.argparse(dbg, [a, b], {})
        return op.build(self, kwargs, dbg)

    def op_power(self, a: Value, b: Value, m: Value | None, dbg: Optional[DebugInfo] = None) -> Value:
        op = PowOp()
        kwargs = op.argparse(dbg, [a, b] + ([m] if m is not None else []), {})
        return op.build(self, kwargs, dbg)

    def op_exp(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = Math_ExpOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_log(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = Math_LogOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_sqrt(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = Math_SqrtOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_abs(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = AbsOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_sign(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = Math_SignOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_int_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = IntCastOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_float_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = FloatCastOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_bool_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        op = BoolCastOp()
        kwargs = op.argparse(dbg, [x], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, IntegerValue)
        return result

    def op_list_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> ListValue:
        op = ListOp()
        kwargs = op.argparse(dbg, [x], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, ListValue)
        return result

    def op_tuple_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> TupleValue:
        op = TupleOp()
        kwargs = op.argparse(dbg, [x], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, TupleValue)
        return result

    def op_set_item(self, the_self: Value, slicing_params: ListValue, the_value: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = SetItemOp()
        kwargs = op.argparse(dbg, [the_self, the_value, slicing_params], {})
        return op.build(self, kwargs, dbg)

    def op_get_item(self, the_self: Value, slicing_params: ListValue, dbg: Optional[DebugInfo] = None) -> Value:
        op = GetItemOp()
        kwargs = op.argparse(dbg, [the_self, slicing_params], {})
        return op.build(self, kwargs, dbg)

    def op_ndarray_set_item(self, the_self: NDArrayValue, slicing_params: ListValue, the_value: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = NDArray_SetItemOp()
        kwargs = op.argparse(dbg, [the_self, the_value, slicing_params], {})
        return op.build(self, kwargs, dbg)

    def op_ndarray_get_item(self, the_self: NDArrayValue, slicing_params: ListValue, dbg: Optional[DebugInfo] = None) -> Value:
        op = NDArray_GetItemOp()
        kwargs = op.argparse(dbg, [the_self, slicing_params], {})
        return op.build(self, kwargs, dbg)

    def op_min(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Value:
        op = MinOp()
        kwargs = op.argparse(dbg, args, {})
        return op.build(self, kwargs, dbg)

    def op_max(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Value:
        op = MaxOp()
        kwargs = op.argparse(dbg, args, {})
        return op.build(self, kwargs, dbg)

    def op_iter(self, x: Value, dbg: Optional[DebugInfo] = None) -> TupleValue:
        op = IterOp()
        kwargs = op.argparse(dbg, [x], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, TupleValue)
        return result

    def op_constant_none(self, dbg: Optional[DebugInfo] = None) -> NoneValue:
        return NoneValue()

    def op_constant_class(self, dt: DTDescriptor, dbg: Optional[DebugInfo] = None) -> ClassValue:
        return ClassValue(dt)

    def op_input(self, indices: Tuple[int, ...], dt: DTDescriptor, kind: str, dbg: Optional[DebugInfo] = None) -> Value:
        op = InputOp(indices, dt, kind)
        return op.build(self, {}, dbg)

    def op_parenthesis(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> TupleValue:
        return TupleValue(tuple(x.type() for x in args), tuple(args))

    def op_square_brackets(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> ListValue:
        return ListValue(list(x.type() for x in args), list(args))

    def op_unary_not(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = NP_LogicalNotOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_unary_sub(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = USubOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_unary_add(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = UAddOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_expose_public(self, x: Value, dbg: Optional[DebugInfo] = None) -> NoneValue:
        op = ExposePublicOp()
        kwargs = op.argparse(dbg, [x], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, NoneValue)
        return result

    def op_poseidon_hash(self, value: Value, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        op = PoseidonHashOp()
        kwargs = op.argparse(dbg, [value], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, IntegerValue)
        return result

    def op_export_external(self, value: Value, for_which: int, key: int | str, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> NoneValue:
        op = ExportExternalOp(for_which, key, indices)
        kwargs = op.argparse(dbg, [value], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, NoneValue)
        return result

    def op_str(self, value: Value, dbg: Optional[DebugInfo] = None) -> StringValue:
        op = StrOp()
        kwargs = op.argparse(dbg, [value], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, StringValue)
        return result

    def op_ndarray_tolist(self, value: NDArrayValue, dbg: Optional[DebugInfo] = None) -> ListValue:
        op = NDArray_ToListOp()
        kwargs = op.argparse(dbg, [value], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, ListValue)
        return result

    def op_ndarray_asarray(self, value: ListValue | TupleValue, dbg: Optional[DebugInfo] = None) -> NDArrayValue:
        op = NP_AsarrayOp()
        kwargs = op.argparse(dbg, [value], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, NDArrayValue)
        return result

    def op_ndarray_astype(self, value: NDArrayValue, dtype: ClassValue, dbg: Optional[DebugInfo] = None) -> NDArrayValue:
        op = NDArray_AsTypeOp()
        kwargs = op.argparse(dbg, [value, dtype], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, NDArrayValue)
        return result

    def op_implicit_type_cast(self, value: Value, dest: DTDescriptor, dbg: Optional[DebugInfo] = None) -> Value:
        op = ImplicitTypeCastOp(dest)
        kwargs = op.argparse(dbg, [value], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, Value)
        return result

    def op_implicit_type_align(self, lhs: Value, rhs: Value, dbg: Optional[DebugInfo] = None) -> TupleValue:
        op = ImplicitTypeAlignOp()
        kwargs = op.argparse(dbg, [lhs, rhs], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, TupleValue)
        return result

    def ir_poseidon_hash(self, values: List[NumberValue], dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = PoseidonHashIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, values, {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_expose_public_i(self, x: IntegerValue, dbg: Optional[DebugInfo] = None) -> NoneValue:
        ir = ExposePublicIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, NoneValue)
        return val

    def ir_expose_public_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> NoneValue:
        ir = ExposePublicFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, NoneValue)
        return val

    def ir_export_external_f(self, value: FloatValue, for_which: int, key: int | str, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> NoneValue:
        ir = ExportExternalFIR(for_which, key, indices)
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [value], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, NoneValue)
        return val

    def ir_export_external_i(self, value: IntegerValue, for_which: int, key: int | str, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> NoneValue:
        ir = ExportExternalIIR(for_which, key, indices)
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [value], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, NoneValue)
        return val

    def ir_invoke_external(
            self,
            external_call_id: int,
            func_name: str,
            args: List[DTDescriptor],
            kwargs: Dict[str, DTDescriptor],
            dbg: Optional[DebugInfo] = None
    ) -> NoneValue:
        ir = InvokeExternalIR(external_call_id, func_name, args, kwargs)
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, NoneValue)
        return val

    def ir_read_integer(self, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = ReadIntegerIR(indices)
        val, stmt = ir.build_ir(len(self.stmts), {}, dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_read_hash(self, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = ReadHashIR(indices)
        val, stmt = ir.build_ir(len(self.stmts), {}, dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_read_float(self, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = ReadFloatIR(indices)
        val, stmt = ir.build_ir(len(self.stmts), {}, dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_constant_int(self, value: int, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = ConstantIntIR(value)
        val, stmt = ir.build_ir(len(self.stmts), {}, dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_constant_float(self, value: float, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = ConstantFloatIR(value)
        val, stmt = ir.build_ir(len(self.stmts), {}, dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_constant_str(self, value: str, dbg: Optional[DebugInfo] = None) -> StringValue:
        ir = ConstantStrIR(value)
        val, stmt = ir.build_ir(len(self.stmts), {}, dbg)
        self.stmts.append(stmt)
        assert isinstance(val, StringValue)
        return val

    def ir_add_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = AddIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_add_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = AddFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_add_str(self, a: StringValue, b: StringValue, dbg: Optional[DebugInfo] = None) -> StringValue:
        ir = AddStrIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, StringValue)
        return val

    def ir_sub_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = SubIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_sub_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = SubFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_mul_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = MulIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_mul_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = MulFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_div_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = DivIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_div_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = DivFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_floor_div_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = FloorDivIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_floor_div_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = FloorDivFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_mod_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = ModIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_mod_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = ModFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_select_i(self, condition: IntegerValue, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = SelectIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [condition, a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_select_f(self, condition: IntegerValue, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = SelectFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [condition, a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_float_cast(self, a: IntegerValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = FloatCastIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_int_cast(self, a: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = IntCastIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_bool_cast(self, a: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = BoolCastIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_abs_i(self, x: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = AbsIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_abs_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = AbsFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_logical_and(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = LogicalAndIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_logical_or(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = LogicalOrIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_logical_not(self, x: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = LogicalNotIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_not_equal_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = NotEqualIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_not_equal_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = NotEqualFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_equal_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = EqualIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_equal_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = EqualFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_equal_hash(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = EqualHashIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_less_than_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = LessThanIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_less_than_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = LessThanFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_less_than_or_equal_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = LessThanOrEqualIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_less_than_or_equal_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = LessThanOrEqualFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_greater_than_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = GreaterThanIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_greater_than_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = GreaterThanFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_greater_than_or_equal_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = GreaterThanOrEqualIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_greater_than_or_equal_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = GreaterThanOrEqualFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [a, b], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_sin_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = SinFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_cos_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = CosFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_tan_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = TanFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_sinh_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = SinHFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_cosh_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = CosHFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_tanh_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = TanHFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_sqrt(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = SqrtFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_exp_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = ExpFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_log_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = LogFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_sign_i(self, x: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = SignIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_sign_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = SignFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_pow_i(self, x: IntegerValue, exponent: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = PowIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x, exponent], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, IntegerValue)
        return val

    def ir_pow_f(self, x: FloatValue, exponent: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        ir = PowFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x, exponent], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, FloatValue)
        return val

    def ir_assert(self, test: IntegerValue, dbg: Optional[DebugInfo] = None) -> NoneValue:
        ir = AssertIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [test], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, NoneValue)
        return val

    def ir_str_i(self, x: IntegerValue, dbg: Optional[DebugInfo] = None) -> StringValue:
        ir = StrIIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, StringValue)
        return val

    def ir_str_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> StringValue:
        ir = StrFIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        assert isinstance(val, StringValue)
        return val

    def ir_print(self, x: StringValue, dbg: Optional[DebugInfo] = None) -> NoneValue:
        ir = PrintIR()
        val, stmt = ir.build_ir(len(self.stmts), ir.argparse(dbg, [x], {}), dbg)
        self.stmts.append(stmt)
        return NoneValue()
