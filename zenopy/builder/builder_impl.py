from typing import List, Dict, Optional, Tuple

from zenopy.builder.ir_builder import IRBuilder
from zenopy.builder.value import Value, ClassValue, TupleValue, ListValue, NoneValue, NDArrayValue, IntegerValue, \
    FloatValue, StringValue, NumberValue
from zenopy.debug.dbg_info import DebugInfo
from zenopy.internal.dt_descriptor import DTDescriptor
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.ir_op.ir_abs_f import AbsFIR
from zenopy.opdef.ir_op.ir_abs_i import AbsIIR
from zenopy.opdef.ir_op.ir_add_f import AddFIR
from zenopy.opdef.ir_op.ir_add_i import AddIIR
from zenopy.opdef.ir_op.ir_assert import AssertIR
from zenopy.opdef.ir_op.ir_bool_cast import BoolCastIR
from zenopy.opdef.ir_op.ir_constant_float import ConstantFloatIR
from zenopy.opdef.ir_op.ir_constant_int import ConstantIntIR
from zenopy.opdef.ir_op.ir_cos_f import CosFIR
from zenopy.opdef.ir_op.ir_cosh_f import CosHFIR
from zenopy.opdef.ir_op.ir_div_f import DivFIR
from zenopy.opdef.ir_op.ir_div_i import DivIIR
from zenopy.opdef.ir_op.ir_eq_f import EqualFIR
from zenopy.opdef.ir_op.ir_eq_i import EqualIIR
from zenopy.opdef.ir_op.ir_exp_f import ExpFIR
from zenopy.opdef.ir_op.ir_export_external_f import ExportExternalFIR
from zenopy.opdef.ir_op.ir_export_external_i import ExportExternalIIR
from zenopy.opdef.ir_op.ir_expose_public_f import ExposePublicFIR
from zenopy.opdef.ir_op.ir_expose_public_i import ExposePublicIIR
from zenopy.opdef.ir_op.ir_float_cast import FloatCastIR
from zenopy.opdef.ir_op.ir_floor_divide_f import FloorDivFIR
from zenopy.opdef.ir_op.ir_floor_divide_i import FloorDivIIR
from zenopy.opdef.ir_op.ir_gt_f import GreaterThanFIR
from zenopy.opdef.ir_op.ir_gt_i import GreaterThanIIR
from zenopy.opdef.ir_op.ir_gte_f import GreaterThanOrEqualFIR
from zenopy.opdef.ir_op.ir_gte_i import GreaterThanOrEqualIIR
from zenopy.opdef.ir_op.ir_hash import HashIR
from zenopy.opdef.ir_op.ir_int_cast import IntCastIR
from zenopy.opdef.ir_op.ir_invoke_external import InvokeExternalIR
from zenopy.opdef.ir_op.ir_log_f import LogFIR
from zenopy.opdef.ir_op.ir_logical_and import LogicalAndIR
from zenopy.opdef.ir_op.ir_logical_not import LogicalNotIR
from zenopy.opdef.ir_op.ir_logical_or import LogicalOrIR
from zenopy.opdef.ir_op.ir_lt_f import LessThanFIR
from zenopy.opdef.ir_op.ir_lt_i import LessThanIIR
from zenopy.opdef.ir_op.ir_lte_f import LessThanOrEqualFIR
from zenopy.opdef.ir_op.ir_lte_i import LessThanOrEqualIIR
from zenopy.opdef.ir_op.ir_mod_f import ModFIR
from zenopy.opdef.ir_op.ir_mod_i import ModIIR
from zenopy.opdef.ir_op.ir_mul_f import MulFIR
from zenopy.opdef.ir_op.ir_mul_i import MulIIR
from zenopy.opdef.ir_op.ir_ne_f import NotEqualFIR
from zenopy.opdef.ir_op.ir_ne_i import NotEqualIIR
from zenopy.opdef.ir_op.ir_pow_f import PowFIR
from zenopy.opdef.ir_op.ir_pow_i import PowIIR
from zenopy.opdef.ir_op.ir_read_float import ReadFloatIR
from zenopy.opdef.ir_op.ir_read_hash import ReadHashIR
from zenopy.opdef.ir_op.ir_read_integer import ReadIntegerIR
from zenopy.opdef.ir_op.ir_select_f import SelectFIR
from zenopy.opdef.ir_op.ir_select_i import SelectIIR
from zenopy.opdef.ir_op.ir_sign_f import SignFIR
from zenopy.opdef.ir_op.ir_sign_i import SignIIR
from zenopy.opdef.ir_op.ir_sin_f import SinFIR
from zenopy.opdef.ir_op.ir_sinh_f import SinHFIR
from zenopy.opdef.ir_op.ir_sqrt_f import SqrtFIR
from zenopy.opdef.ir_op.ir_sub_f import SubFIR
from zenopy.opdef.ir_op.ir_sub_i import SubIIR
from zenopy.opdef.ir_op.ir_tan_f import TanFIR
from zenopy.opdef.ir_op.ir_tanh_f import TanHFIR
from zenopy.opdef.ndarray.op_get_item import NDArray_GetItemOp
from zenopy.opdef.ndarray.op_set_item import NDArray_SetItemOp
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.opdef.nocls.op_abs import AbsOp
from zenopy.opdef.nocls.op_add import AddOp
from zenopy.opdef.nocls.op_assert import AssertOp
from zenopy.opdef.nocls.op_bool_cast import BoolCastOp
from zenopy.opdef.nocls.op_bool_scalar import BoolScalarOp
from zenopy.opdef.nocls.op_div import DivOp
from zenopy.opdef.nocls.op_eq import EqualOp
from zenopy.opdef.nocls.op_exp import ExpOp
from zenopy.opdef.nocls.op_export_external import ExportExternalOp
from zenopy.opdef.nocls.op_expose_public import ExposePublicOp
from zenopy.opdef.nocls.op_float_cast import FloatCastOp
from zenopy.opdef.nocls.op_float_scalar import FloatScalarOp
from zenopy.opdef.nocls.op_floor_divide import FloorDivideOp
from zenopy.opdef.nocls.op_gt import GreaterThanOp
from zenopy.opdef.nocls.op_gte import GreaterThanOrEqualOp
from zenopy.opdef.nocls.op_hash import HashOp
from zenopy.opdef.nocls.op_input import InputOp
from zenopy.opdef.nocls.op_int_cast import IntCastOp
from zenopy.opdef.nocls.op_integer_scalar import IntegerScalarOp
from zenopy.opdef.nocls.op_iter import IterOp
from zenopy.opdef.nocls.op_list import ListOp
from zenopy.opdef.nocls.op_log import LogOp
from zenopy.opdef.nocls.op_lt import LessThanOp
from zenopy.opdef.nocls.op_lte import LessThanOrEqualOp
from zenopy.opdef.nocls.op_mat_mul import MatMulOp
from zenopy.opdef.nocls.op_max import MaxOp
from zenopy.opdef.nocls.op_min import MinOp
from zenopy.opdef.nocls.op_mod import ModOp
from zenopy.opdef.nocls.op_mul import MulOp
from zenopy.opdef.nocls.op_ne import NotEqualOp
from zenopy.opdef.nocls.op_not import NotOp
from zenopy.opdef.nocls.op_pow import PowOp
from zenopy.opdef.nocls.op_select import SelectOp
from zenopy.opdef.nocls.op_get_item import GetItemOp
from zenopy.opdef.nocls.op_set_item import SetItemOp
from zenopy.opdef.nocls.op_sign import SignOp
from zenopy.opdef.nocls.op_sqrt import SqrtOp
from zenopy.opdef.nocls.op_sub import SubOp
from zenopy.opdef.nocls.op_tuple import TupleOp
from zenopy.opdef.nocls.op_uadd import UAddOp
from zenopy.opdef.nocls.op_usub import USubOp


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
        op = ExpOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_log(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = LogOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_sqrt(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = SqrtOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_abs(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = AbsOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_sign(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = SignOp()
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

    def op_bool_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        op = BoolCastOp()
        kwargs = op.argparse(dbg, [x], {})
        return op.build(self, kwargs, dbg)

    def op_float_scalar(self, x: Value, dbg: Optional[DebugInfo] = None) -> FloatValue:
        op = FloatScalarOp()
        kwargs = op.argparse(dbg, [x], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, FloatValue)
        return result

    def op_int_scalar(self, x: Value, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        op = IntegerScalarOp()
        kwargs = op.argparse(dbg, [x], {})
        result = op.build(self, kwargs, dbg)
        assert isinstance(result, IntegerValue)
        return result

    def op_bool_scalar(self, x: Value, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        op = BoolScalarOp()
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

    def op_constant_string(self, value: str, dbg: Optional[DebugInfo] = None) -> StringValue:
        return StringValue(value)

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
        op = NotOp()
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

    def op_hash(self, value: Value, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        op = HashOp()
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

    def ir_hash(self, values: List[NumberValue], dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = HashIR()
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

    def ir_invoke_external(self, external_call_id: int, dbg: Optional[DebugInfo] = None) -> NoneValue:
        ir = InvokeExternalIR(external_call_id)
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

    def ir_read_hash(self, input_id: int, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        ir = ReadHashIR(0, input_id)
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
