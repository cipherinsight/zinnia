from zinnia.op_def.np_like.op_allclose import NP_AllCloseOp


class NP_IsCloseOp(NP_AllCloseOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.isclose"

    @classmethod
    def get_name(cls) -> str:
        return "isclose"
