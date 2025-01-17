from inspect import isclass
from typing import Optional

from pyzk.opdef.ndarray.op_T import NDArray_TOp
from pyzk.opdef.ndarray.op_argmax import NDArray_ArgMaxOp
from pyzk.opdef.ndarray.op_argmin import NDArray_ArgMinOp
from pyzk.opdef.ndarray.op_flat import NDArray_FlatOp
from pyzk.opdef.ndarray.op_max import NDArray_MaxOp
from pyzk.opdef.ndarray.op_min import NDArray_MinOp
from pyzk.opdef.ndarray.op_reshape import NDArray_ReshapeOp
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
from pyzk.opdef.nocls.op_add import AddOp
from pyzk.opdef.nocls.op_all import AllOp
from pyzk.opdef.nocls.op_and import AndOp
from pyzk.opdef.nocls.op_any import AnyOp
from pyzk.opdef.nocls.op_assert import AssertOp
from pyzk.opdef.nocls.op_set_item import SetItemOp
from pyzk.opdef.nocls.op_bool_cast import BoolCastOp
from pyzk.opdef.nocls.op_concatenate import ConcatenateOp
from pyzk.opdef.nocls.op_constant_cast import ConstantCastOp
from pyzk.opdef.nocls.op_cos import CosOp
from pyzk.opdef.nocls.op_cosh import CosHOp
from pyzk.opdef.nocls.op_div import DivOp
from pyzk.opdef.nocls.op_eq import EqualOp
from pyzk.opdef.nocls.op_exp import ExpOp
from pyzk.opdef.nocls.op_float_cast import FloatCastOp
from pyzk.opdef.nocls.op_floor_divide import FloorDivideOp
from pyzk.opdef.nocls.op_gt import GreaterThanOp
from pyzk.opdef.nocls.op_gte import GreaterThanOrEqualOp
from pyzk.opdef.nocls.op_input import InputOp
from pyzk.opdef.nocls.op_int_cast import IntCastOp
from pyzk.opdef.nocls.op_len import LenOp
from pyzk.opdef.nocls.op_list import ListOp
from pyzk.opdef.nocls.op_log import LogOp
from pyzk.opdef.nocls.op_logical_and import LogicalAndOp
from pyzk.opdef.nocls.op_logical_or import LogicalOrOp
from pyzk.opdef.nocls.op_lt import LessThanOp
from pyzk.opdef.nocls.op_lte import LessThanOrEqualOp
from pyzk.opdef.nocls.op_mat_mul import MatMulOp
from pyzk.opdef.nocls.op_maximum import MaximumOp
from pyzk.opdef.nocls.op_minimum import MinimumOp
from pyzk.opdef.nocls.op_mod import ModOp
from pyzk.opdef.nocls.op_mul import MulOp
from pyzk.opdef.nocls.op_ne import NotEqualOp
from pyzk.opdef.nocls.op_not import NotOp
from pyzk.opdef.nocls.op_or import OrOp
from pyzk.opdef.nocls.op_pow import PowOp
from pyzk.opdef.nocls.op_range import RangeOp
from pyzk.opdef.nocls.op_select import SelectOp
from pyzk.opdef.nocls.op_sign import SignOp
from pyzk.opdef.nocls.op_sin import SinOp
from pyzk.opdef.nocls.op_sinh import SinHOp
from pyzk.opdef.nocls.op_get_item import GetItemOp
from pyzk.opdef.nocls.op_sqrt import SqrtOp
from pyzk.opdef.nocls.op_stack import StackOp
from pyzk.opdef.nocls.op_sub import SubOp
from pyzk.opdef.nocls.op_tan import TanOp
from pyzk.opdef.nocls.op_tanh import TanHOp
from pyzk.opdef.nocls.op_tuple import TupleOp
from pyzk.opdef.nocls.op_usub import USubOp


class Operators:
    class NoCls:
        ADD = AddOp
        AND = AndOp
        ASSERT = AssertOp
        ASSIGN_SLICE = SetItemOp
        BOOL_CAST = BoolCastOp
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
        MAT_MUL = MatMulOp
        MUL = MulOp
        OR = OrOp
        RANGE = RangeOp
        SLICE = GetItemOp
        SUB = SubOp
        TUPLE = TupleOp
        USUB = USubOp
        CONSTANT_CAST = ConstantCastOp
        INPUT = InputOp
        NOT = NotOp
        SELECT = SelectOp
        FLOAT_CAST = FloatCastOp
        INT_CAST = IntCastOp
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
        MOD = ModOp
        MAXIMUM = MaximumOp
        MINIMUM = MinimumOp
        FLOOR_DIV = FloorDivideOp
        ABS = AbsOp
        SIGN = SignOp
        CONCATENATE = ConcatenateOp
        STACK = StackOp
        ANY = AnyOp
        ALL = AllOp

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
        FLAT = NDArray_FlatOp
        RESHAPE = NDArray_ReshapeOp

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
