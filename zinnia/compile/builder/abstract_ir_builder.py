from typing import Optional, List, Dict, Tuple

from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.value import Value, NumberValue, IntegerValue, FloatValue, ListValue, TupleValue, NDArrayValue, \
    NoneValue, ClassValue, StringValue
from zinnia.compile.type_sys import DTDescriptor


class AbsIRBuilderInterface:
    def __init__(self):
        pass

    def create_op(self, operator, args: List[Value], kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def create_ir(self, operator, args: List[Value], kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_select(self, condition: Value, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_assert(self, test: Value, condition: IntegerValue | None, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_less_than(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_less_than_or_equal(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_equal(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_not_equal(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_greater_than(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_greater_than_or_equal(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_add(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_subtract(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_multiply(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_divide(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_floor_divide(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_mat_mul(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_modulo(self, a: Value, b: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_power(self, a: Value, b: Value, m: Value | None, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_exp(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_log(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_sqrt(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_abs(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_sign(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_int_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_float_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_bool_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_float_scalar(self, x: Value, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def op_int_scalar(self, x: Value, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def op_bool_scalar(self, x: Value, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def op_list_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> ListValue:
        raise NotImplementedError()

    def op_tuple_cast(self, x: Value, dbg: Optional[DebugInfo] = None) -> TupleValue:
        raise NotImplementedError()

    def op_set_item(self, the_self: Value, slicing_params: ListValue, the_value: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_get_item(self, the_self: Value, slicing_params: ListValue, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_ndarray_set_item(self, the_self: NDArrayValue, slicing_params: ListValue, the_value: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_ndarray_get_item(self, the_self: NDArrayValue, slicing_params: ListValue, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_min(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_max(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_iter(self, x: Value, dbg: Optional[DebugInfo] = None) -> TupleValue:
        raise NotImplementedError()

    def op_constant_none(self, dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()

    def op_constant_class(self, dt: DTDescriptor, dbg: Optional[DebugInfo] = None) -> ClassValue:
        raise NotImplementedError()

    def op_input(self, indices: Tuple[int, ...], dt: DTDescriptor, kind: str, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_parenthesis(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> TupleValue:
        raise NotImplementedError()

    def op_square_brackets(self, args: List[Value], dbg: Optional[DebugInfo] = None) -> ListValue:
        raise NotImplementedError()

    def op_unary_not(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_unary_sub(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_unary_add(self, x: Value, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()

    def op_expose_public(self, value: Value, dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()

    def op_export_external(self, value: Value, for_which: int, key: int | str, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()

    def op_poseidon_hash(self, value: Value, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def op_str(self, value: Value, dbg: Optional[DebugInfo] = None) -> StringValue:
        raise NotImplementedError()

    def op_ndarray_tolist(self, value: Value, dbg: Optional[DebugInfo] = None) -> ListValue:
        raise NotImplementedError()

    def ir_expose_public_i(self, value: IntegerValue, dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()

    def ir_expose_public_f(self, value: FloatValue, dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()

    def ir_export_external_i(self, value: IntegerValue, for_which: int, key: int | str, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()

    def ir_export_external_f(self, value: FloatValue, for_which: int, key: int | str, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()

    def ir_invoke_external(
            self,
            external_call_id: int,
            func_name: str,
            args: List[DTDescriptor],
            kwargs: Dict[str, DTDescriptor],
            dbg: Optional[DebugInfo] = None
    ) -> NoneValue:
        raise NotImplementedError()

    def ir_poseidon_hash(self, values: List[NumberValue], dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_read_integer(self, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_read_hash(self, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_read_float(self, indices: Tuple[int, ...], dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_assert(self, test: IntegerValue, dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()

    def ir_constant_int(self, value: int, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_constant_float(self, value: float, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_constant_str(self, value: str, dbg: Optional[DebugInfo] = None) -> StringValue:
        raise NotImplementedError()

    def ir_add_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_add_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_add_str(self, a: StringValue, b: StringValue, dbg: Optional[DebugInfo] = None) -> StringValue:
        raise NotImplementedError()

    def ir_sub_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_sub_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_mul_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_mul_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_div_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_div_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_floor_div_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_floor_div_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_mod_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_mod_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_select_i(self, condition: IntegerValue, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_select_f(self, condition: IntegerValue, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_float_cast(self, a: IntegerValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_int_cast(self, a: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_bool_cast(self, a: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_abs_i(self, x: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_abs_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_logical_and(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_logical_or(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_logical_not(self, a: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_not_equal_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_not_equal_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_equal_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_equal_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_equal_hash(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_less_than_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_less_than_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_less_than_or_equal_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_less_than_or_equal_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_greater_than_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_greater_than_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_greater_than_or_equal_i(self, a: IntegerValue, b: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_greater_than_or_equal_f(self, a: FloatValue, b: FloatValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_sin_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_cos_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_tan_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_sinh_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_cosh_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_tanh_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_sqrt(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_exp_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_log_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_sign_i(self, x: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_sign_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_pow_i(self, x: IntegerValue, exponent: IntegerValue, dbg: Optional[DebugInfo] = None) -> IntegerValue:
        raise NotImplementedError()

    def ir_pow_f(self, x: FloatValue, exponent: FloatValue, dbg: Optional[DebugInfo] = None) -> FloatValue:
        raise NotImplementedError()

    def ir_str_i(self, x: IntegerValue, dbg: Optional[DebugInfo] = None) -> StringValue:
        raise NotImplementedError()

    def ir_str_f(self, x: FloatValue, dbg: Optional[DebugInfo] = None) -> StringValue:
        raise NotImplementedError()

    def ir_print(self, x: StringValue, dbg: Optional[DebugInfo] = None) -> NoneValue:
        raise NotImplementedError()
