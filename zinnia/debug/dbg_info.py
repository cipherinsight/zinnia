class DebugInfo:
    def __init__(self, method_name: str, source_code: str, circuit: bool, lineno: int, col_offset: int, end_lineno: int, end_col_offset: int):
        self.method_name = method_name
        self.source_code = source_code
        self.circuit = circuit
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset
