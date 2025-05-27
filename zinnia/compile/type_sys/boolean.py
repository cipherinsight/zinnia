from typing import Tuple, Optional, Dict, Any, List

from zinnia.compile.type_sys import IntegerDTDescriptor
from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo


class BooleanDTDescriptor(IntegerDTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls) -> str:
        return "Boolean"

    @classmethod
    def get_alise_typenames(cls) -> List[str]:
        return ["Boolean", "bool", "Bool", "boolean"]

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'BooleanDTDescriptor':
        return BooleanDTDescriptor()

    def export(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'BooleanDTDescriptor':
        return BooleanDTDescriptor()
