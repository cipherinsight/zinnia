from typing import List, Dict

from zenopy.internal.dt_descriptor import DTDescriptor


class ExternalCall:
    def __init__(self, call_id: int, method_name: str, args: List[DTDescriptor], kwargs: Dict[str, DTDescriptor]):
        self.call_id = call_id
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
