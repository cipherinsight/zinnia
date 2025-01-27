from typing import Any, Dict

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor


class HashedDTDescriptor(DTDescriptor):
    def __init__(self, dtype: DTDescriptor):
        super().__init__()
        self.dtype = dtype

    @classmethod
    def get_typename(cls):
        return f"Hashed"

    def __str__(self) -> str:
        return f'{self.get_typename()}[{self.dtype}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.dtype == other.dtype

    def export(self) -> Dict[str, Any]:
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return {
            "dtype": DTDescriptorFactory.export(self.dtype)
        }

    @staticmethod
    def import_from(data: Dict) -> 'HashedDTDescriptor':
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        dtype = DTDescriptorFactory.import_from(data['dtype'])
        return HashedDTDescriptor(dtype)
