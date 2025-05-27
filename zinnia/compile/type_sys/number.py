from typing import Optional, Tuple, Any, Dict

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo


class NumberDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    def __new__(cls, *args, **kwargs):
        if cls is DTDescriptor:
            raise TypeError(f"<NumberDTDescriptor> must be subclassed.")
        return object.__new__(cls)

    @classmethod
    def get_typename(cls):
        return "Number"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'NumberDTDescriptor':
        raise NotImplementedError()

    def export(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def import_from(data: Dict) -> 'NumberDTDescriptor':
        return NumberDTDescriptor()
