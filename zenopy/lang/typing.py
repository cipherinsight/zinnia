class _ZKTyping:
    def __init__(self, kind: str):
        self.kind = kind

    def __getitem__(self, item):
        pass

Private = _ZKTyping("Private")
Public = _ZKTyping("Public")
Hashed = _ZKTyping("Hashed")
