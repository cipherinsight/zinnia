from .lang.type import Integer, Float, Complex, Tuple, List, NDArray, DynamicNDArray, PoseidonHashed, integer, Int, Bool, Boolean, boolean, Array, DynArray, Vec
from .lang.typing import Private, Public, private, public
from .lang.operator import *
from .api.zk_chip import ZKChip, zk_chip
from .api.zk_external_func import ZKExternalFunc, zk_external
from .api.zk_circuit import ZKCircuit, zk_circuit
from .spec import requires
from .api.zk_compiled_program import ZKCompiledProgram
from .api.zk_program_input import ZKProgramInput
from .exec.exec_result import ZKExecResult
from .debug.exception import ZinniaException
from .config.zinnia_config import ZinniaConfig
from .lang.namespace import NamespaceNP


np = NamespaceNP()
try:
    import numpy as np
except ImportError:
    pass
