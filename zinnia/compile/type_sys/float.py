from typing import Optional, Tuple, Dict, Any

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.compile.type_sys.number import NumberDTDescriptor
from zinnia.debug.dbg_info import DebugInfo


class FloatDTDescriptor(NumberDTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "Float"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'FloatDTDescriptor':
        return FloatDTDescriptor()

    def export(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'FloatDTDescriptor':
        return FloatDTDescriptor()
