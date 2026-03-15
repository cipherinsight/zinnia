from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import BooleanValue, IntegerValue, ListValue, NDArrayValue, TupleValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class MerkleVerifyOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "merkle_verify"

    @classmethod
    def get_name(cls) -> str:
        return "merkle_verify"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("leaf"),
            AbstractOp._ParamEntry("root"),
            AbstractOp._ParamEntry("siblings"),
            AbstractOp._ParamEntry("directions"),
        ]

    @staticmethod
    def _as_seq(x: Value) -> List[Value] | None:
        if isinstance(x, ListValue):
            return x.values()
        if isinstance(x, TupleValue):
            return list(x.values())
        if isinstance(x, NDArrayValue):
            if len(x.shape()) != 1:
                return None
            return x.flattened_values()
        return None

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        leaf = kwargs["leaf"]
        root = kwargs["root"]
        siblings = kwargs["siblings"]
        directions = kwargs["directions"]

        if not isinstance(leaf, IntegerValue) or not isinstance(root, IntegerValue):
            raise TypeInferenceError(dbg, "`leaf` and `root` of `merkle_verify` must be Integer")

        sibling_values = self._as_seq(siblings)
        direction_values = self._as_seq(directions)
        if sibling_values is None:
            raise TypeInferenceError(dbg, "`siblings` of `merkle_verify` must be a List, Tuple, or 1-D NDArray")
        if direction_values is None:
            raise TypeInferenceError(dbg, "`directions` of `merkle_verify` must be a List, Tuple, or 1-D NDArray")
        if len(sibling_values) != len(direction_values):
            raise TypeInferenceError(dbg, "`siblings` and `directions` of `merkle_verify` must have the same length")

        acc = leaf
        for sibling, direction in zip(sibling_values, direction_values):
            if not isinstance(sibling, IntegerValue):
                raise TypeInferenceError(dbg, "Each element in `siblings` of `merkle_verify` must be Integer")
            if not isinstance(direction, (BooleanValue, IntegerValue)):
                raise TypeInferenceError(dbg, "Each element in `directions` of `merkle_verify` must be Bool or Integer")

            is_right = builder.op_bool_cast(direction, dbg)
            left = builder.op_select(is_right, sibling, acc, dbg)
            right = builder.op_select(is_right, acc, sibling, dbg)
            assert isinstance(left, IntegerValue)
            assert isinstance(right, IntegerValue)
            acc = builder.ir_poseidon_hash([left, right], dbg)

        result = builder.op_equal(acc, root, dbg)
        assert isinstance(result, BooleanValue)
        return result
