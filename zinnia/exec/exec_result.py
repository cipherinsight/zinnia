class ZKExecResult:
    def __init__(self, satisfied: bool, public_outputs=None, proof=None):
        self.satisfied = satisfied
        self.public_outputs = public_outputs or {}
        self.proof = proof  # Optional ZKProofResult, None for mock execution

    def is_satisfied(self) -> bool:
        return self.satisfied

    def __bool__(self) -> bool:
        return self.satisfied

    def __repr__(self):
        return f"ZKExecResult(satisfied={self.satisfied})"
