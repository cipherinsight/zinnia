from typing import Tuple, Dict, Any, Optional, List

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidAnnotationException


class TupleDTDescriptor(DTDescriptor):
    def __init__(self, elements_type: Tuple[DTDescriptor, ...]):
        super().__init__()
        self.elements_type = elements_type

    def __str__(self) -> str:
        return f'{self.get_typename()}[{", ".join([str(x) for x in self.elements_type])}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.elements_type == other.elements_type

    @classmethod
    def get_typename(cls):
        return "Tuple"

    @classmethod
    def get_alise_typenames(cls) -> List[str]:
        return ["Tuple", "tuple"]

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'TupleDTDescriptor':
        if len(args) == 0:
            raise InvalidAnnotationException(dbg_i, "Annotation `Tuple` requires 1 or more arguments")
        return TupleDTDescriptor(tuple(arg for arg in args))

    def export(self) -> Dict[str, Any]:
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return {
            'elements': [
                DTDescriptorFactory.export(element) for element in self.elements_type
            ]
        }

    @staticmethod
    def import_from(data: Dict) -> 'TupleDTDescriptor':
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        elements_type = [DTDescriptorFactory.import_from(dtype) for dtype in data['elements']]
        return TupleDTDescriptor(tuple(elements_type))
