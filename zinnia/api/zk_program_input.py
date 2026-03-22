from typing import Dict


class ZKProgramInput:
    class Kind:
        PUBLIC = "Public"
        PRIVATE = "Private"

    def __init__(self, name: str, dt: dict, kind: str):
        self.name = name
        self.dt = dt
        self.kind = kind

    def get_name(self) -> str:
        return self.name

    def get_dt(self) -> dict:
        return self.dt

    def get_kind(self) -> str:
        return self.kind

    def is_public(self) -> bool:
        return self.kind == ZKProgramInput.Kind.PUBLIC

    def is_private(self) -> bool:
        return self.kind == ZKProgramInput.Kind.PRIVATE

    def export(self) -> Dict:
        return {
            "name": self.name,
            "dt": self.dt,
            "kind": self.kind,
        }

    @staticmethod
    def import_from(data: Dict) -> 'ZKProgramInput':
        return ZKProgramInput(
            name=data['name'],
            dt=data['dt'],
            kind=data['kind'],
        )
