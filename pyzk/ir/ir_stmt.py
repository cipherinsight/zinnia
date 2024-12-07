from typing import Dict

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.annotation import Annotation
from pyzk.debug.dbg_info import DebugInfo


class IRStatement:
    def __init__(
        self,
        stmt_id: int,
        operator: AbstractOp,
        arguments: Dict[str, int],
        annotation: Annotation | None = None,
        dbg_i: DebugInfo | None = None,
    ):
        self.stmt_id = stmt_id
        self.operator = operator
        self.arguments = arguments
        self.annotation = annotation
        self.dbg_i = dbg_i
        assert all([arg is not None for arg in arguments])
