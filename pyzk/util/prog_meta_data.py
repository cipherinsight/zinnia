from typing import List, Tuple


class ProgramInputMetadata:
    def __init__(self, typename: str, shape: Tuple[int, ...], public: bool):
        self.typename = typename
        self.shape = shape
        self.public = public


class ProgramMetadata:
    def __init__(self):
        self.inputs = []

    def set_program_inputs(self, inputs: List[ProgramInputMetadata]):
        self.inputs = inputs