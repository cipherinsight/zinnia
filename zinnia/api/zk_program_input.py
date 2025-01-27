from typing import Dict

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor


class ZKProgramInput:
    class Kind:
        PUBLIC = "Public"
        PRIVATE = "Private"
        HASHED = "Hashed"

    def __init__(self, name: str, dt: DTDescriptor, kind: str):
        self.name = name
        self.dt = dt
        self.kind = kind

    def get_name(self) -> str:
        return self.name

    def get_dt(self) -> DTDescriptor:
        return self.dt

    def get_kind(self) -> str:
        return self.kind

    def is_public(self) -> bool:
        return self.kind == ZKProgramInput.Kind.PUBLIC

    def is_private(self) -> bool:
        return self.kind == ZKProgramInput.Kind.PRIVATE

    def is_hashed(self) -> bool:
        return self.kind == ZKProgramInput.Kind.HASHED

    def export(self) -> Dict:
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return {
            "name": self.name,
            "dt": DTDescriptorFactory.export(self.dt),
            "kind": self.kind,
        }

    @staticmethod
    def import_from(data: Dict) -> 'ZKProgramInput':
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return ZKProgramInput(
            name=data['name'],
            dt=DTDescriptorFactory.import_from(data['dt']),
            kind=data['kind'],
        )
