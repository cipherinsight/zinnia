from typing import Optional

from zinnia.op_def.abstract.abstract_op import AbstractOp
import zinnia.op_def.np_like as np_ops
import zinnia.op_def.math as math_ops
import zinnia.op_def.nocls as global_ops
import zinnia.op_def.ndarray as ndarray_ops
import zinnia.op_def.lst as list_ops
import zinnia.op_def.tupl as tuple_ops


class Operators:
    NoCls = [
        global_ops.TupleOp, global_ops.StrOp, global_ops.RangeOp, global_ops.PrintOp,
        global_ops.PowOp, global_ops.MinOp, global_ops.MaxOp, global_ops.RangeOp,
        global_ops.ListOp, global_ops.LenOp, global_ops.FloatCastOp, global_ops.BoolCastOp,
        global_ops.IntCastOp, global_ops.SumOp, global_ops.AnyOp, global_ops.AllOp
    ]
    NDArray = [
        ndarray_ops.NDArray_ProdOp, ndarray_ops.NDArray_SumOp, ndarray_ops.NDArray_TOp, ndarray_ops.NDArray_TransposeOp,
        ndarray_ops.NDArray_ToListOp, ndarray_ops.NDArray_AsTypeOp, ndarray_ops.NDArray_MaxOp, ndarray_ops.NDArray_MinOp,
        ndarray_ops.NDArray_ArgMaxOp, ndarray_ops.NDArray_ArgMinOp, ndarray_ops.NDArray_ShapeOp, ndarray_ops.NDArray_ReshapeOp,
        ndarray_ops.NDArray_FlatOp, ndarray_ops.NDArray_DtypeOp, ndarray_ops.NDArray_AllOp, ndarray_ops.NDArray_AnyOp,
        ndarray_ops.NDArray_NdimOp, ndarray_ops.NDArray_RepeatOp, ndarray_ops.NDArray_SizeOp, ndarray_ops.NDArray_FlattenOp
    ]
    Tuple = [
        tuple_ops.Tuple_CountOp, tuple_ops.Tuple_IndexOp
    ]
    List = [
        list_ops.List_AppendOp, list_ops.List_ExtendOp, list_ops.List_InsertOp, list_ops.List_PopOp,
        list_ops.List_CopyOp, list_ops.List_ClearOp, list_ops.List_ReverseOp, list_ops.List_IndexOp,
        list_ops.List_CountOp, list_ops.List_RemoveOp,
    ]
    String = []
    NPLike = [
        np_ops.NP_EyeOp, np_ops.NP_ZerosOp, np_ops.NP_OnesOp, np_ops.NP_IdentityOp,
        np_ops.NP_ConcatenateOp, np_ops.NP_ConcatOp, np_ops.NP_StackOp, np_ops.NP_MinimumOp,
        np_ops.NP_MaximumOp, np_ops.NP_LogicalNotOp, np_ops.NP_LogicalAndOp, np_ops.NP_LogicalOrOp,
        np_ops.NP_AsarrayOp, np_ops.NP_AbsOp, np_ops.NP_AbsoluteOp, np_ops.NP_ACosOp,
        np_ops.NP_AddOp, np_ops.NP_AllOp, np_ops.NP_AllCloseOp, np_ops.NP_AMaxOp,
        np_ops.NP_AMinOp, np_ops.NP_AnyOp, np_ops.NP_ArgmaxOp, np_ops.NP_ArgminOp,
        np_ops.NP_ArrayEqualOp, np_ops.NP_ArrayEquivOp, np_ops.NP_ASinOp, np_ops.NP_ATanOp,
        np_ops.NP_CosOp, np_ops.NP_CosHOp, np_ops.NP_DivideOp, np_ops.NP_EqualOp,
        np_ops.NP_ExpOp, np_ops.NP_FAbsOp, np_ops.NP_FloorDivideOp, np_ops.NP_FMaxOp,
        np_ops.NP_FMinOp, np_ops.NP_FModOp, np_ops.NP_GreaterOp, np_ops.NP_GreaterEqualOp,
        np_ops.NP_IsCloseOp, np_ops.NP_LessOp, np_ops.NP_LessEqualOp, np_ops.NP_LogOp,
        np_ops.NP_LogicalXorOp, np_ops.NP_MaxOp, np_ops.NP_ModOp, np_ops.NP_MultiplyOp,
        np_ops.NP_NegativeOp, np_ops.NP_PositiveOp, np_ops.NP_NotEqualOp, np_ops.NP_PowerOp,
        np_ops.NP_PowOp, np_ops.NP_ProdOp, np_ops.NP_SignOp, np_ops.NP_SinHOp,
        np_ops.NP_SqrtOp, np_ops.NP_SubtractOp, np_ops.NP_SumOp, np_ops.NP_TanOp,
        np_ops.NP_TanHOp, np_ops.NP_RepeatOp, np_ops.NP_SizeOp, np_ops.NP_AppendOp,
        np_ops.NP_DotOp, np_ops.NP_ARangeOp, np_ops.NP_LinspaceOp, np_ops.NP_ArrayOp
    ]
    Zinnia = [
        *NPLike
    ]
    Math = [
        math_ops.Math_TanHOp, math_ops.Math_TanOp, math_ops.Math_LogOp, math_ops.Math_ExpOp,
        math_ops.Math_CosOp, math_ops.Math_SinOp, math_ops.Math_CosHOp, math_ops.Math_SinHOp,
        math_ops.Math_SqrtOp, math_ops.Math_FAbsOp, math_ops.Math_InvOp
    ]

    @staticmethod
    def get_namespaces():
        return ["NDArray", "Tuple", "List", "String", "np", "zinnia", "math"]

    @staticmethod
    def get_operator(operator_name: str, class_name: Optional[str]):
        lookup = {
            None: Operators.NoCls,
            "NDArray": Operators.NDArray,
            "Tuple": Operators.Tuple,
            "List": Operators.List,
            "String": Operators.String,
            "np": Operators.NPLike,
            "zinnia": Operators.Zinnia,
            "math": Operators.Math,
        }
        ops = lookup.get(class_name)
        for op in ops:
            assert issubclass(op, AbstractOp)
            if op.get_name() == operator_name:
                return op
        return None

    @staticmethod
    def instantiate_operator(operator_name: str, class_name: Optional[str], *args, **kwargs):
        op = Operators.get_operator(operator_name, class_name)
        if op is None:
            return None
        return op(*args, **kwargs)
