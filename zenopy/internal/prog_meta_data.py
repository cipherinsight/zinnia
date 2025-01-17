from typing import List, Dict

from zenopy.internal.dt_descriptor import DTDescriptor


class ProgramInputMetadata:
    def __init__(self, dt: DTDescriptor, name: str, public: bool):
        self.dt = dt
        self.name = name
        self.public = public


class ProgramMetadata:
    def __init__(self) -> None:
        self.inputs = []
        self.circuit_name = "circuit_func"

    def set_program_inputs(self, inputs: List[ProgramInputMetadata]) -> None:
        self.inputs = inputs

    def set_circuit_name(self, circuit_name: str) -> None:
        self.circuit_name = circuit_name

    def export(self) -> Dict:
        return {
            "inputs": [{
                "dt": inp.dt.export(),
                "public": inp.public,
                "name": inp.name,
            } for inp in self.inputs],
        }
