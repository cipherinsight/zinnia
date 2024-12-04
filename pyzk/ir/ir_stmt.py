from typing import Dict

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.util.annotation import Annotation
from pyzk.util.source_pos_info import SourcePosInfo


class IRStatement:
    def __init__(
        self,
        stmt_id: int,
        operator: AbstractOp,
        arguments: Dict[str, int],
        annotation: Annotation | None = None,
        spi: SourcePosInfo | None = None,
    ):
        self.stmt_id = stmt_id
        self.operator = operator
        self.arguments = arguments
        self.annotation = annotation
        self.spi = spi
        assert all([arg is not None for arg in arguments])
