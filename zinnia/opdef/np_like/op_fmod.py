from zinnia.opdef.np_like.op_mod import NP_ModOp


class NP_FModOp(NP_ModOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.fmod"

    @classmethod
    def get_name(cls) -> str:
        return "fmod"
