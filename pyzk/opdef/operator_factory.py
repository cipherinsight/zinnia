from inspect import isclass
from typing import Optional

from pyzk.opdef.ndarray.op_T import NDArray_TOp
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
from pyzk.opdef.nocls.op_add import AddOp
from pyzk.opdef.nocls.op_and import AndOp
from pyzk.opdef.nocls.op_assert import AssertOp
from pyzk.opdef.nocls.op_assign_slice import AssignSliceOp
from pyzk.opdef.nocls.op_bool_cast import BoolCastOp
from pyzk.opdef.nocls.op_constant_cast import ConstantCastOp
from pyzk.opdef.nocls.op_constant import ConstantOp
from pyzk.opdef.nocls.op_div import DivOp
from pyzk.opdef.nocls.op_eq import EqualOp
from pyzk.opdef.nocls.op_gt import GreaterThanOp
from pyzk.opdef.nocls.op_gte import GreaterThanOrEqualOp
from pyzk.opdef.nocls.op_input import InputOp
from pyzk.opdef.nocls.op_len import LenOp
from pyzk.opdef.nocls.op_list import ListOp
from pyzk.opdef.nocls.op_logical_and import LogicalAndOp
from pyzk.opdef.nocls.op_logical_or import LogicalOrOp
from pyzk.opdef.nocls.op_lt import LessThanOp
from pyzk.opdef.nocls.op_lte import LessThanOrEqualOp
from pyzk.opdef.nocls.op_mat_mul import MatMulOp
from pyzk.opdef.nocls.op_mul import MulOp
from pyzk.opdef.nocls.op_ne import NotEqualOp
from pyzk.opdef.nocls.op_not import NotOp
from pyzk.opdef.nocls.op_or import OrOp
from pyzk.opdef.nocls.op_parenthesis import ParenthesisOp
from pyzk.opdef.nocls.op_range import RangeOp
from pyzk.opdef.nocls.op_read_number import ReadNumberOp
from pyzk.opdef.nocls.op_slice import SliceOp
from pyzk.opdef.nocls.op_square_brackets import SquareBracketsOp
from pyzk.opdef.nocls.op_sub import SubOp
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
        EQ = EqualOp
        GT = GreaterThanOp
        GTE = GreaterThanOrEqualOp
        LEN = LenOp
        LIST = ListOp
        LOGICAL_AND = LogicalAndOp
        LOGICAL_OR = LogicalOrOp
        LT = LessThanOp
        LTE = LessThanOrEqualOp
        MAT_MUL = MatMulOp
        MUL = MulOp
        NE = NotEqualOp
        OR = OrOp
        PARENTHESIS = ParenthesisOp
        RANGE = RangeOp
        READ_NUMBER = ReadNumberOp
        SLICE = SliceOp
        SQUARE_BRACKETS = SquareBracketsOp
        SUB = SubOp
        TUPLE = TupleOp
        USUB = USubOp
        CONSTANT_CAST = ConstantCastOp
        INPUT = InputOp
        NOT = NotOp

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
        raise NotImplementedError(f'Operator {class_name if class_name is not None else ""}::{operator_name} not found')

    @staticmethod
    def instantiate_operator(operator_name: str, class_name: Optional[str], *args, **kwargs):
        op = Operators.get_operator(operator_name, class_name)
        return op(*args, **kwargs)
