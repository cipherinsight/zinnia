from typing import Dict, Any, List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import OperatorCallError
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.triplet import Value


class AbstractOp:
    class _ParamEntry:
        def __init__(self, name: str, default: bool = False):
            self.name = name
            self.default = default

    def __init__(self):
        pass

    def get_signature(self) -> str:
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def is_inplace(cls) -> bool:
        return False

    @classmethod
    def requires_condition(cls) -> bool:
        return False

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def get_param_entries(self) -> List[_ParamEntry]:
        raise NotImplementedError()

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params = self.get_param_entries()
        should_have_default = False
        for entry in params:
            if entry.default:
                should_have_default = True
            elif should_have_default:
                raise ValueError("Internal Error: Should have defaults here")
        mapping = {}
        filled = []
        for param in params:
            mapping[param.name] = None
        for i, arg in enumerate(args):
            if i >= len(params):
                raise OperatorCallError(dbg_i, f"Too many arguments for operator `{self.get_name()}`")
            mapping[params[i].name] = arg
            filled.append(params[i].name)
        for k, v in kwargs.items():
            if k in filled:
                raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` got multiple values for argument `{k}`")
            if k not in mapping.keys():
                raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` got an unexpected keyword argument `{k}`")
            mapping[k] = v
            filled.append(k)
        for param in params:
            if param.name not in filled and not param.default:
                raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` missing required argument `{param.name}`")
        parse_result = {}
        for k, v in mapping.items():
            if v is not None:
                parse_result[k] = v
        return parse_result

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()
