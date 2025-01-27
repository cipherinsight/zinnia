from typing import Optional, Tuple, Dict, Any

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo


class ClassDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "Class"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'ClassDTDescriptor':
        return ClassDTDescriptor()

    def export(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'ClassDTDescriptor':
        return ClassDTDescriptor()
