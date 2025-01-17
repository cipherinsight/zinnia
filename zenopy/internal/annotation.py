from zenopy.internal.dt_descriptor import DTDescriptor


class Annotation:
    def __init__(
        self,
        dt: DTDescriptor,
        public: bool = False,
    ):
        self.dt = dt
        self.public = public
