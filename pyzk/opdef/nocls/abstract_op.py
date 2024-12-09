from typing import Dict, Any, List, Optional

from pyzk.debug.exception import OperatorCallError
from pyzk.internal.dt_descriptor import DTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


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

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def dce_keep(self) -> bool:
        return False

    def get_param_entries(self) -> List[_ParamEntry]:
        raise NotImplementedError()

    def params_parse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
                raise OperatorCallError(dbg_i, f"Too many arguments for operator `{self.get_signature()}`")
            mapping[params[i].name] = arg
            filled.append(params[i].name)
        for k, v in kwargs.items():
            if k in filled:
                raise OperatorCallError(dbg_i, f"Operator `{self.get_signature()}` got multiple values for argument `{k}`")
            if k not in mapping.keys():
                raise OperatorCallError(dbg_i, f"Operator `{self.get_signature()}` got an unexpected keyword argument `{k}`")
            mapping[k] = v
            filled.append(k)
        for param in params:
            if param.name not in filled and not param.default:
                raise OperatorCallError(dbg_i, f"Operator `{self.get_signature()}` missing required argument `{param.name}`")
        return mapping

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        raise NotImplementedError()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        raise NotImplementedError()
