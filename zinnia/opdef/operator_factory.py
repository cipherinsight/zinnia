from inspect import isclass
from typing import Optional

from zinnia.opdef.ndarray.op_T import NDArray_TOp
from zinnia.opdef.ndarray.op_argmax import NDArray_ArgMaxOp
from zinnia.opdef.ndarray.op_argmin import NDArray_ArgMinOp
from zinnia.opdef.ndarray.op_asarray import NDArray_AsarrayOp
from zinnia.opdef.ndarray.op_astype import NDArray_AsTypeOp
from zinnia.opdef.ndarray.op_dtype import NDArray_DtypeOp
from zinnia.opdef.ndarray.op_flat import NDArray_FlatOp
from zinnia.opdef.ndarray.op_max import NDArray_MaxOp
from zinnia.opdef.ndarray.op_min import NDArray_MinOp
from zinnia.opdef.ndarray.op_reshape import NDArray_ReshapeOp
from zinnia.opdef.ndarray.op_tolist import NDArray_ToListOp
from zinnia.opdef.ndarray.op_transpose import NDArray_TransposeOp
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.opdef.ndarray.op_all import NDArray_AllOp
from zinnia.opdef.ndarray.op_any import NDArray_AnyOp
from zinnia.opdef.ndarray.op_eye import NDArray_EyeOp
from zinnia.opdef.ndarray.op_identity import NDArray_IdentityOp
from zinnia.opdef.ndarray.op_ones import NDArray_OnesOp
from zinnia.opdef.ndarray.op_shape import NDArray_ShapeOp
from zinnia.opdef.ndarray.op_sum import NDArray_SumOp
from zinnia.opdef.ndarray.op_zeros import NDArray_ZerosOp
from zinnia.opdef.nocls.op_abs import AbsOp
from zinnia.opdef.nocls.op_add import AddOp
from zinnia.opdef.nocls.op_all import AllOp
from zinnia.opdef.nocls.op_and import AndOp
from zinnia.opdef.nocls.op_any import AnyOp
from zinnia.opdef.nocls.op_assert import AssertOp
from zinnia.opdef.nocls.op_print import PrintOp
from zinnia.opdef.nocls.op_set_item import SetItemOp
from zinnia.opdef.nocls.op_bool_cast import BoolCastOp
from zinnia.opdef.nocls.op_concatenate import ConcatenateOp
from zinnia.opdef.nocls.op_constant_cast import ConstantCastOp
from zinnia.opdef.nocls.op_cos import CosOp
from zinnia.opdef.nocls.op_cosh import CosHOp
from zinnia.opdef.nocls.op_div import DivOp
from zinnia.opdef.nocls.op_eq import EqualOp
from zinnia.opdef.nocls.op_exp import ExpOp
from zinnia.opdef.nocls.op_float_cast import FloatCastOp
from zinnia.opdef.nocls.op_floor_divide import FloorDivideOp
from zinnia.opdef.nocls.op_gt import GreaterThanOp
from zinnia.opdef.nocls.op_gte import GreaterThanOrEqualOp
from zinnia.opdef.nocls.op_input import InputOp
from zinnia.opdef.nocls.op_int_cast import IntCastOp
from zinnia.opdef.nocls.op_len import LenOp
from zinnia.opdef.nocls.op_list import ListOp
from zinnia.opdef.nocls.op_log import LogOp
from zinnia.opdef.nocls.op_logical_and import LogicalAndOp
from zinnia.opdef.nocls.op_logical_or import LogicalOrOp
from zinnia.opdef.nocls.op_lt import LessThanOp
from zinnia.opdef.nocls.op_lte import LessThanOrEqualOp
from zinnia.opdef.nocls.op_mat_mul import MatMulOp
from zinnia.opdef.nocls.op_maximum import MaximumOp
from zinnia.opdef.nocls.op_minimum import MinimumOp
from zinnia.opdef.nocls.op_mod import ModOp
from zinnia.opdef.nocls.op_mul import MulOp
from zinnia.opdef.nocls.op_ne import NotEqualOp
from zinnia.opdef.nocls.op_not import NotOp
from zinnia.opdef.nocls.op_or import OrOp
from zinnia.opdef.nocls.op_pow import PowOp
from zinnia.opdef.nocls.op_range import RangeOp
from zinnia.opdef.nocls.op_select import SelectOp
from zinnia.opdef.nocls.op_sign import SignOp
from zinnia.opdef.nocls.op_sin import SinOp
from zinnia.opdef.nocls.op_sinh import SinHOp
from zinnia.opdef.nocls.op_get_item import GetItemOp
from zinnia.opdef.nocls.op_sqrt import SqrtOp
from zinnia.opdef.nocls.op_stack import StackOp
from zinnia.opdef.nocls.op_str import StrOp
from zinnia.opdef.nocls.op_sub import SubOp
from zinnia.opdef.nocls.op_sum import SumOp
from zinnia.opdef.nocls.op_tan import TanOp
from zinnia.opdef.nocls.op_tanh import TanHOp
from zinnia.opdef.nocls.op_tuple import TupleOp
from zinnia.opdef.nocls.op_usub import USubOp


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
        STR = StrOp
        PRINT = PrintOp
        SUM = SumOp

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
        ASARRAY = NDArray_AsarrayOp
        TOLIST = NDArray_ToListOp
        DTYPE = NDArray_DtypeOp
        ASTYPE = NDArray_AsTypeOp

    class Tuple:
        pass

    class List:
        pass

    class String:
        pass

    @staticmethod
    def get_operator(operator_name: str, class_name: Optional[str]):
        lookup = {
            None: Operators.NoCls,
            "NDArray": Operators.NDArray,
            "Tuple": Operators.Tuple,
            "List": Operators.List,
            "String": Operators.String,
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
