from typing import List, Dict, Optional, Any

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NoneValue, StringValue, IntegerValue


class PrintOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "print"

    @classmethod
    def get_name(cls) -> str:
        return "print"

    @classmethod
    def is_inplace(cls) -> bool:
        return True

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        parsed_dict = {}
        for key in kwargs.keys():
            if key not in ['sep', 'end', 'flush']:
                raise TypeInferenceError(dbg_i, f"Unexpected keyword argument `{key}` for `print`")
            parsed_dict[key] = kwargs[key]
        for i, arg in enumerate(args):
            parsed_dict[f"_x_{i}"] = arg
        return parsed_dict

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        sep = kwargs.get('sep', builder.ir_constant_str(" "))
        end = kwargs.get('end', builder.ir_constant_str("\n"))
        flush = kwargs.get('flush', builder.ir_constant_int(0))
        if not isinstance(sep, StringValue):
            raise TypeInferenceError(dbg, f"Expected string for `sep` but got {sep.type()}")
        if not isinstance(end, StringValue):
            raise TypeInferenceError(dbg, f"Expected string for `end` but got {end.type()}")
        if not isinstance(flush, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected integer (boolean) for `flush` but got {flush.type()}")
        args = []
        for k, v in kwargs.get_kwargs().items():
            if k not in ['sep', 'end', 'flush']:
                args.append(v)
        print_value = builder.ir_constant_str("")
        for i, arg in enumerate(args):
            print_value = builder.ir_add_str(print_value, builder.op_str(arg))
            if i < len(args) - 1:
                print_value = builder.ir_add_str(print_value, sep)
        print_value = builder.ir_add_str(print_value, end)
        builder.ir_print(kwargs.get_condition(), print_value)
        return NoneValue()
