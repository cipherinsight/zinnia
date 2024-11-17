from typing import List, Tuple

from pyzk.util.annotation import Annotation
from pyzk.util.op_name import OpName
from pyzk.util.source_pos_info import SourcePosInfo


class IRStatement:
    def __init__(
        self,
        stmt_id: int,
        op_name: str,
        op_args: List[int],
        constant_value: int | None = None,
        slicing_args: List[int | Tuple[int, int]] = None,
        slicing_assign_args: List[List[int | Tuple[int, int]]] = None,
        constant_args: List[int] | None = None,
        annotation: Annotation | None = None,
        source_pos_info: SourcePosInfo | None = None,
    ):
        self.stmt_id = stmt_id
        self.op = op_name
        self.args = op_args
        self.slicing_args = slicing_args
        self.slicing_assign_args = slicing_assign_args
        self.constant_value = constant_value
        self.constant_args = constant_args
        self.annotation = annotation
        self.source_pos_info = source_pos_info
        assert (op_name == OpName.Special.CONSTANT and self.constant_value is not None) or self.constant_value is None
        assert (op_name == OpName.Special.SLICING_ASSIGN and len(self.slicing_assign_args) > 0) or self.slicing_assign_args is None
        assert (op_name == OpName.Special.SLICING and len(self.slicing_args) > 0) or self.slicing_args is None
