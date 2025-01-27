from typing import List, Dict, Any, Optional

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidAnnotationException


class ListDTDescriptor(DTDescriptor):
    def __init__(self, elements_dtype: List[DTDescriptor]):
        super().__init__()
        self.elements_dtype = elements_dtype

    def __str__(self) -> str:
        return f'{self.get_typename()}[{", ".join([str(x) for x in self.elements_dtype])}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.elements_dtype == other.elements_dtype

    @classmethod
    def get_typename(cls):
        return "List"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: List[DTDescriptor | int]) -> 'ListDTDescriptor':
        if len(args) == 0:
            raise InvalidAnnotationException(dbg_i, "Annotation `List` requires 1 or more arguments")
        for arg in args:
            if not isinstance(arg, DTDescriptor):
                raise InvalidAnnotationException(dbg_i, "Annotation `List` requires all type arguments to be a datatype")
        return ListDTDescriptor([arg for arg in args])

    def export(self) -> Dict[str, Any]:
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return {
            'elements': [
                DTDescriptorFactory.export(element) for element in self.elements_dtype
            ]
        }

    @staticmethod
    def import_from(data: Dict) -> 'ListDTDescriptor':
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        elements_dtype = [DTDescriptorFactory.import_from(dtype) for dtype in data['elements']]
        return ListDTDescriptor(list(elements_dtype))
