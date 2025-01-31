from typing import Optional

from zinnia.opdef.abstract.abstract_op import AbstractOp
import zinnia.opdef.np_like as np_ops
import zinnia.opdef.math as math_ops
import zinnia.opdef.nocls as global_ops
import zinnia.opdef.ndarray as ndarray_ops


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
        ndarray_ops.NDArray_NdimOp
    ]
    Tuple = []
    List = []
    String = []
    NPLike = [
        np_ops.NP_EyeOp, np_ops.NP_ZerosOp, np_ops.NP_OnesOp, np_ops.NP_IdentityOp,
        np_ops.NP_ConcatenateOp, np_ops.NP_ConcatOp, np_ops.NP_StackOp, np_ops.NP_MinimumOp,
        np_ops.NP_MaximumOp, np_ops.NP_LogicalNotOp, np_ops.NP_LogicalAndOp, np_ops.NP_LogicalOrOp,
        np_ops.NP_AsarrayOp
    ]
    Zinnia = [
        *NPLike
    ]
    Math = [
        math_ops.Math_TanHOp, math_ops.Math_TanOp, math_ops.Math_LogOp, math_ops.Math_ExpOp,
        math_ops.Math_CosOp, math_ops.Math_SinOp, math_ops.Math_CosHOp, math_ops.Math_SinHOp,
        math_ops.Math_SqrtOp
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
