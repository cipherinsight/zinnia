class ZKExecResult:
    def __init__(self, satisfied: bool):
        self.satisfied = satisfied

    def is_satisfied(self) -> bool:
        return self.satisfied

    def __bool__(self):
        return self.satisfied
