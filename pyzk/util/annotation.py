from typing import Tuple


class Annotation:
    def __init__(
        self,
        typename: str | None = None,
        shape: Tuple[int, ...] | None = None,
        public: bool | None = None,
    ):
        self.typename = typename
        self.shape = shape
        self.public = public
        assert typename is not None and shape is not None and public is not None
