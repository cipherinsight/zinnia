from inspect import isclass
from typing import Optional

from pyzk.opdef.ndarray.op_T import NDArray_TOp
from pyzk.opdef.ndarray.op_argmax import NDArray_ArgMaxOp
from pyzk.opdef.ndarray.op_argmin import NDArray_ArgMinOp
from pyzk.opdef.ndarray.op_max import NDArray_MaxOp
from pyzk.opdef.ndarray.op_min import NDArray_MinOp
from pyzk.opdef.ndarray.op_transpose import NDArray_TransposeOp
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.opdef.ndarray.op_all import NDArray_AllOp
from pyzk.opdef.ndarray.op_any import NDArray_AnyOp
from pyzk.opdef.ndarray.op_eye import NDArray_EyeOp
from pyzk.opdef.ndarray.op_identity import NDArray_IdentityOp
from pyzk.opdef.ndarray.op_ones import NDArray_OnesOp
from pyzk.opdef.ndarray.op_shape import NDArray_ShapeOp
from pyzk.opdef.ndarray.op_sum import NDArray_SumOp
from pyzk.opdef.ndarray.op_zeros import NDArray_ZerosOp
from pyzk.opdef.nocls.op_abs import AbsOp
from pyzk.opdef.nocls.op_abs_f import AbsFOp
from pyzk.opdef.nocls.op_abs_i import AbsIOp
from pyzk.opdef.nocls.op_add import AddOp
from pyzk.opdef.nocls.op_add_f import AddFOp
from pyzk.opdef.nocls.op_add_i import AddIOp
from pyzk.opdef.nocls.op_and import AndOp
from pyzk.opdef.nocls.op_assert import AssertOp
from pyzk.opdef.nocls.op_assign_slice import AssignSliceOp
from pyzk.opdef.nocls.op_bool_cast import BoolCastOp
from pyzk.opdef.nocls.op_constant_cast import ConstantCastOp
from pyzk.opdef.nocls.op_constant import ConstantOp
from pyzk.opdef.nocls.op_constant_class import ConstantClassOp
from pyzk.opdef.nocls.op_constant_float import ConstantFloatOp
from pyzk.opdef.nocls.op_constant_none import ConstantNoneOp
from pyzk.opdef.nocls.op_cos import CosOp
from pyzk.opdef.nocls.op_cosh import CosHOp
from pyzk.opdef.nocls.op_div import DivOp
from pyzk.opdef.nocls.op_div_f import DivFOp
from pyzk.opdef.nocls.op_div_i import DivIOp
from pyzk.opdef.nocls.op_eq import EqualOp
from pyzk.opdef.nocls.op_eq_f import EqualFOp
from pyzk.opdef.nocls.op_eq_i import EqualIOp
from pyzk.opdef.nocls.op_exp import ExpOp
from pyzk.opdef.nocls.op_float import FloatOp
from pyzk.opdef.nocls.op_floor_divide import FloorDivideOp
from pyzk.opdef.nocls.op_gt import GreaterThanOp
from pyzk.opdef.nocls.op_gt_f import GreaterThanFOp
from pyzk.opdef.nocls.op_gt_i import GreaterThanIOp
from pyzk.opdef.nocls.op_gte import GreaterThanOrEqualOp
from pyzk.opdef.nocls.op_gte_f import GreaterThanOrEqualFOp
from pyzk.opdef.nocls.op_gte_i import GreaterThanOrEqualIOp
from pyzk.opdef.nocls.op_input import InputOp
from pyzk.opdef.nocls.op_int import IntOp
from pyzk.opdef.nocls.op_len import LenOp
from pyzk.opdef.nocls.op_list import ListOp
from pyzk.opdef.nocls.op_log import LogOp
from pyzk.opdef.nocls.op_logical_and import LogicalAndOp
from pyzk.opdef.nocls.op_logical_or import LogicalOrOp
from pyzk.opdef.nocls.op_lt import LessThanOp
from pyzk.opdef.nocls.op_lt_f import LessThanFOp
from pyzk.opdef.nocls.op_lt_i import LessThanIOp
from pyzk.opdef.nocls.op_lte import LessThanOrEqualOp
from pyzk.opdef.nocls.op_lte_f import LessThanOrEqualFOp
from pyzk.opdef.nocls.op_lte_i import LessThanOrEqualIOp
from pyzk.opdef.nocls.op_mat_mul import MatMulOp
from pyzk.opdef.nocls.op_maximum import MaximumOp
from pyzk.opdef.nocls.op_minimum import MinimumOp
from pyzk.opdef.nocls.op_mod import ModOp
from pyzk.opdef.nocls.op_mod_f import ModFOp
from pyzk.opdef.nocls.op_mod_i import ModIOp
from pyzk.opdef.nocls.op_mul import MulOp
from pyzk.opdef.nocls.op_mul_f import MulFOp
from pyzk.opdef.nocls.op_mul_i import MulIOp
from pyzk.opdef.nocls.op_ne import NotEqualOp
from pyzk.opdef.nocls.op_ne_f import NotEqualFOp
from pyzk.opdef.nocls.op_ne_i import NotEqualIOp
from pyzk.opdef.nocls.op_not import NotOp
from pyzk.opdef.nocls.op_or import OrOp
from pyzk.opdef.nocls.op_parenthesis import ParenthesisOp
from pyzk.opdef.nocls.op_pow import PowOp
from pyzk.opdef.nocls.op_pow_f import PowFOp
from pyzk.opdef.nocls.op_pow_i import PowIOp
from pyzk.opdef.nocls.op_range import RangeOp
from pyzk.opdef.nocls.op_read_float import ReadFloatOp
from pyzk.opdef.nocls.op_read_integer import ReadIntegerOp
from pyzk.opdef.nocls.op_select import SelectOp
from pyzk.opdef.nocls.op_sign import SignOp
from pyzk.opdef.nocls.op_sign_f import SignFOp
from pyzk.opdef.nocls.op_sign_i import SignIOp
from pyzk.opdef.nocls.op_sin import SinOp
from pyzk.opdef.nocls.op_sinh import SinHOp
from pyzk.opdef.nocls.op_slice import SliceOp
from pyzk.opdef.nocls.op_sqrt import SqrtOp
from pyzk.opdef.nocls.op_square_brackets import SquareBracketsOp
from pyzk.opdef.nocls.op_sub import SubOp
from pyzk.opdef.nocls.op_sub_f import SubFOp
from pyzk.opdef.nocls.op_sub_i import SubIOp
from pyzk.opdef.nocls.op_tan import TanOp
from pyzk.opdef.nocls.op_tanh import TanHOp
from pyzk.opdef.nocls.op_tuple import TupleOp
from pyzk.opdef.nocls.op_usub import USubOp


class Operators:
    class NoCls:
        ADD = AddOp
        AND = AndOp
        ASSERT = AssertOp
        ASSIGN_SLICE = AssignSliceOp
        BOOL_CAST = BoolCastOp
        CONSTANT = ConstantOp
        DIV = DivOp
        LEN = LenOp
        LIST = ListOp
        LOGICAL_AND = LogicalAndOp
        LOGICAL_OR = LogicalOrOp
        EQ = EqualOp
        NE = NotEqualOp
        LT = LessThanOp
        LTE = LessThanOrEqualOp
        GT = GreaterThanOp
        GTE = GreaterThanOrEqualOp
        EQ_I = EqualIOp
        NE_I = NotEqualIOp
        LT_I = LessThanIOp
        LTE_I = LessThanOrEqualIOp
        GT_I = GreaterThanIOp
        GTE_I = GreaterThanOrEqualIOp
        EQ_F = EqualFOp
        NE_F = NotEqualFOp
        LT_F = LessThanFOp
        LTE_F = LessThanOrEqualFOp
        GT_F = GreaterThanFOp
        GTE_F = GreaterThanOrEqualFOp
        MAT_MUL = MatMulOp
        MUL = MulOp
        OR = OrOp
        PARENTHESIS = ParenthesisOp
        RANGE = RangeOp
        READ_INTEGER = ReadIntegerOp
        READ_FLOAT = ReadFloatOp
        SLICE = SliceOp
        SQUARE_BRACKETS = SquareBracketsOp
        SUB = SubOp
        TUPLE = TupleOp
        USUB = USubOp
        CONSTANT_CAST = ConstantCastOp
        INPUT = InputOp
        NOT = NotOp
        CONSTANT_NONE = ConstantNoneOp
        CONSTANT_CLASS = ConstantClassOp
        SELECT = SelectOp
        ADD_I = AddIOp
        ADD_F = AddFOp
        SUB_I = SubIOp
        SUB_F = SubFOp
        MUL_I = MulIOp
        MUL_F = MulFOp
        DIV_I = DivIOp
        DIV_F = DivFOp
        FLOAT_CAST = FloatOp
        INT_CAST = IntOp
        CONSTANT_FLOAT = ConstantFloatOp
        SIN = SinOp
        COS = CosOp
        TAN = TanOp
        SINH = SinHOp
        COSH = CosHOp
        TANH = TanHOp
        LOG = LogOp
        EXP = ExpOp
        SQRT = SqrtOp
        POW = PowOp
        POW_I = PowIOp
        POW_F = PowFOp
        MOD = ModOp
        MOD_I = ModIOp
        MOD_F = ModFOp
        MAXIMUM = MaximumOp
        MINIMUM = MinimumOp
        FLOOR_DIV = FloorDivideOp
        ABS = AbsOp
        ABS_I = AbsIOp
        ABS_F = AbsFOp
        SIGN = SignOp
        SIGN_I = SignIOp
        SIGN_F = SignFOp

    class NDArray:
        ALL = NDArray_AllOp
        ANY = NDArray_AnyOp
        EYE = NDArray_EyeOp
        IDENTITY = NDArray_IdentityOp
        ONES = NDArray_OnesOp
        SHAPE = NDArray_ShapeOp
        SUM = NDArray_SumOp
        ZEROS = NDArray_ZerosOp
        T = NDArray_TOp
        TRANSPOSE = NDArray_TransposeOp
        MIN = NDArray_MinOp
        MAX = NDArray_MaxOp
        ARGMIN = NDArray_ArgMinOp
        ARGMAX = NDArray_ArgMaxOp

    class Tuple:
        pass

    @staticmethod
    def get_operator(operator_name: str, class_name: Optional[str]):
        lookup = {
            None: Operators.NoCls,
            "NDArray": Operators.NDArray,
            "Tuple": Operators.Tuple,
        }
        ops = lookup.get(class_name)
        for k, op in ops.__dict__.items():
            if not isclass(op) or not issubclass(op, AbstractOp):
                continue
            if op.get_name() == operator_name and class_name is None:
                return op
            elif f"{class_name}::{operator_name}" == op.get_name() and class_name is not None:
                return op
        return None

    @staticmethod
    def instantiate_operator(operator_name: str, class_name: Optional[str], *args, **kwargs):
        op = Operators.get_operator(operator_name, class_name)
        if op is None:
            return None
        return op(*args, **kwargs)
