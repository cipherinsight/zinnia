from typing import Tuple, Optional, Any, Dict, Type, List

from zinnia.debug.exception.transforming import InvalidAnnotationException
from zinnia.debug.dbg_info import DebugInfo


class DTDescriptor(object):
    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if cls is DTDescriptor:
            raise TypeError(f"<DTDescriptor> must be subclassed.")
        return object.__new__(cls)

    def __str__(self) -> str:
        return self.get_typename()

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__ and self.get_typename() == other.get_typename()

    def export(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def import_from(data: Dict) -> 'DTDescriptor':
        raise NotImplementedError()

    @classmethod
    def get_typename(cls):
        raise NotImplementedError()

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[Any, ...]) -> 'DTDescriptor':
        raise NotImplementedError()

    def is_t(self, other: Type) -> bool:
        return isinstance(self, other)
