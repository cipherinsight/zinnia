from typing import Dict, Any, List, Optional

from pyzk.exception.contextual import OperatorCallError
from pyzk.util.dt_descriptor import DTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


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

    def dce_keep(self) -> bool:
        return False

    def get_param_entries(self) -> List[_ParamEntry]:
        raise NotImplementedError()

    def params_parse(self, spi: Optional[SourcePosInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
                raise OperatorCallError(spi, f"Too many arguments for operator `{self.get_signature()}`")
            mapping[params[i].name] = arg
            filled.append(params[i].name)
        for k, v in kwargs.items():
            if k in filled:
                raise OperatorCallError(spi, f"Operator `{self.get_signature()}` got multiple values for argument `{k}`")
            if k not in mapping.keys():
                raise OperatorCallError(spi, f"Operator `{self.get_signature()}` got an unexpected keyword argument `{k}`")
            mapping[k] = v
            filled.append(k)
        for param in params:
            if param.name not in filled and not param.default:
                raise OperatorCallError(spi, f"Operator `{self.get_signature()}` missing required argument `{param.name}`")
        return mapping

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        raise NotImplementedError()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        raise NotImplementedError()
