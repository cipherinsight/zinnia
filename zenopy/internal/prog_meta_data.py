from typing import List, Dict, Tuple

from zenopy.internal.dt_descriptor import DTDescriptor, IntegerType, FloatType


class ProgramInputMetadata:
    def __init__(self, dt: DTDescriptor, name: str, kind: str):
        self.dt = dt
        self.name = name
        self.kind = kind

    def export(self):
        return {
            "dt": self.dt.export(),
            "kind": self.kind,
            "name": self.name,
        }


class ProgramCompiledInputMetadata:
    def __init__(self, dt: DTDescriptor, indices: Tuple[int, ...]):
        self.dt = dt
        self.indices = indices
        assert self.dt == IntegerType or self.dt == FloatType


class ProgramMetadata:
    def __init__(self) -> None:
        self.inputs = []
        self.compiled_inputs = []
        self.circuit_name = "circuit_func"

    def set_program_inputs(self, inputs: List[ProgramInputMetadata]) -> None:
        self.inputs = inputs

    def set_program_compiled_inputs(self, compiled_inputs: List[ProgramCompiledInputMetadata]) -> None:
        self.compiled_inputs = compiled_inputs

    def set_circuit_name(self, circuit_name: str) -> None:
        self.circuit_name = circuit_name
