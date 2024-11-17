from typing import List

from pyzk.util.op_name import OpName


class ConstantFold:
    @staticmethod
    def constant_fold(op_name: str, args: List[int]) -> int | None:
        if op_name == OpName.Binary.ADD:
            return args[0] + args[1]
        elif op_name == OpName.Binary.SUB:
            return args[0] - args[1]
        elif op_name == OpName.Binary.MUL:
            return args[0] * args[1]
        elif op_name == OpName.Unary.USUB:
            return -args[0]
        else:
            return None
