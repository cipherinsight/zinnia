from typing import Tuple, Optional, Dict, Any, List

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.compile.type_sys.number import NumberDTDescriptor
from zinnia.debug.dbg_info import DebugInfo


class IntegerDTDescriptor(NumberDTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls) -> str:
        return "Integer"

    @classmethod
    def get_alise_typenames(cls) -> List[str]:
        return ["Integer", "int", "Int", "integer"]

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'IntegerDTDescriptor':
        return IntegerDTDescriptor()

    def export(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'IntegerDTDescriptor':
        return IntegerDTDescriptor()
