from typing import List

from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.np_like import NP_AsarrayOp


class NP_ArrayOp(NP_AsarrayOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.array"

    @classmethod
    def get_name(cls) -> str:
        return "array"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("val"),
            AbstractOp._ParamEntry("dtype", default=True)
        ]
