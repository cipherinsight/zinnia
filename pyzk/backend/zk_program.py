class ZKProgram:
    def __init__(self, circuit_name: str, source: str):
        self.circuit_name = circuit_name
        self.source = source


class Halo2ZKProgram(ZKProgram):
    def __init__(self, circuit_name: str, source: str):
        super().__init__(circuit_name, source)
