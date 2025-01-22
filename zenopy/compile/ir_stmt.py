from typing import List, Optional

from zenopy.debug.dbg_info import DebugInfo


class IRStatement:
    def __init__(
        self,
        stmt_id: int,
        ir_operator,
        arguments: List[int],
        dbg: Optional[DebugInfo] = None,
    ):
        self.stmt_id = stmt_id
        self.operator = ir_operator
        self.arguments = arguments
        self.dbg = dbg
        assert all([arg is not None for arg in arguments])

    def __copy__(self):
        return IRStatement(self.stmt_id, self.operator, self.arguments.copy(), self.dbg)
