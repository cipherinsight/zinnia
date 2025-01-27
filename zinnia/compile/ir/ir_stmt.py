from typing import List, Optional, Dict

from zinnia.debug.dbg_info import DebugInfo


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

    def export(self) -> Dict:
        from zinnia.opdef.ir_factory import IRFactory

        return {
            "stmt_id": self.stmt_id,
            "operator": IRFactory.export(self.operator),
            "arguments": self.arguments,
        }

    @staticmethod
    def import_from(data: Dict):
        from zinnia.opdef.ir_factory import IRFactory

        return IRStatement(
            data["stmt_id"],
            IRFactory.import_from(data["operator"]),
            data["arguments"],
        )
