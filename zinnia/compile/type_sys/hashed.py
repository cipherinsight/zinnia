from typing import Any, Dict, Optional, List

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidAnnotationException


class PoseidonHashedDTDescriptor(DTDescriptor):
    def __init__(self, dtype: DTDescriptor):
        super().__init__()
        self.dtype = dtype

    @classmethod
    def get_typename(cls):
        return f"PoseidonHashed"

    def __str__(self) -> str:
        return f'{self.get_typename()}[{self.dtype}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.dtype == other.dtype

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: List[DTDescriptor | int]) -> 'PoseidonHashedDTDescriptor':
        if len(args) != 1:
            raise InvalidAnnotationException(dbg_i, "Annotation `PoseidonHashed` requires exactly 1 argument")
        if not isinstance(args[0], DTDescriptor):
            raise InvalidAnnotationException(dbg_i, "Annotation `List` requires all type arguments to be a datatype")
        return PoseidonHashedDTDescriptor(args[0])

    def export(self) -> Dict[str, Any]:
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return {
            "dtype": DTDescriptorFactory.export(self.dtype)
        }

    @staticmethod
    def import_from(data: Dict) -> 'PoseidonHashedDTDescriptor':
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        dtype = DTDescriptorFactory.import_from(data['dtype'])
        return PoseidonHashedDTDescriptor(dtype)
