from typing import List, Dict

from pyzk.internal.dt_descriptor import DTDescriptor


class ProgramInputMetadata:
    def __init__(self, dt: DTDescriptor, public: bool):
        self.dt = dt
        self.public = public


class ProgramMetadata:
    def __init__(self) -> None:
        self.inputs = []

    def set_program_inputs(self, inputs: List[ProgramInputMetadata]) -> None:
        self.inputs = inputs

    def export(self) -> Dict:
        return {
            "inputs": [{
                "dt": inp.dt.export(),
                "public": inp.public,
            } for inp in self.inputs],
        }
