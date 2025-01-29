from typing import Optional, Tuple, Dict, Any, List

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo


class StringDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls) -> str:
        return "String"

    @classmethod
    def get_alise_typenames(cls) -> List[str]:
        return ["String", "str"]

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'DTDescriptor':
        return StringDTDescriptor()

    def export(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'StringDTDescriptor':
        return StringDTDescriptor()
