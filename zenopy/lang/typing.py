class _ZKTyping:
    def __init__(self, private: bool):
        self.private = private

    def __getitem__(self, item):
        pass

Private = _ZKTyping(True)
Public = _ZKTyping(False)
