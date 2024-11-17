class SourcePosInfo:
    def __init__(self, lineno: int, col_offset: int, end_lineno: int, end_col_offset: int):
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset
