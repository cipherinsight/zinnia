from typing import Any

from zinnia.config.base import ConfigBase
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.config.optimization_config import OptimizationConfig


class ZinniaConfig(ConfigBase):
    BACKEND_HALO2 = "halo2"
    BACKEND_CIRCOM = "circom"
    BACKEND_NOIR = "noir"
    DEFAULT_RECURSION_LIMIT = 100
    DEFAULT_LOOP_LIMIT = 1000

    def __init__(
            self,
            backend: str = BACKEND_HALO2,
            recursion_limit: int = DEFAULT_RECURSION_LIMIT,
            loop_limit: int = DEFAULT_LOOP_LIMIT,
            mock_config: MockExecConfig = MockExecConfig(),
            optimization_config: OptimizationConfig = OptimizationConfig()
    ):
        super().__init__()
        self.set("backend", backend)
        self.set("recursion_limit", recursion_limit)
        self.set("loop_limit", loop_limit)
        self.set("mock_config", mock_config)
        self.set("optimization_config", optimization_config)

    def verify(self, key: str, value: Any) -> Any:
        if key == "backend":
            if value not in [self.BACKEND_HALO2, self.BACKEND_CIRCOM, self.BACKEND_NOIR]:
                raise ValueError(f"Invalid `backend` specified: {value}")
            return value
        elif key == "mock_config":
            if isinstance(value, MockExecConfig):
                return value
            elif isinstance(value, dict):
                return MockExecConfig().deserialize(value)
            raise ValueError(f"Invalid `mock_config` specified: {value}")
        elif key == "optimization_config":
            if isinstance(value, OptimizationConfig):
                return value
            elif isinstance(value, dict):
                return OptimizationConfig().deserialize(value)
            raise ValueError(f"Invalid `optimization_config` specified: {value}")
        elif key == "recursion_limit":
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid `recursion_limit` specified: {value}")
            return value
        elif key == "loop_limit":
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid `loop_limit` specified: {value}")
            return value
        return value

    def get_backend(self) -> str:
        return self.get("backend")

    def mock_config(self) -> MockExecConfig:
        return self.get("mock_config")

    def optimization_config(self) -> OptimizationConfig:
        return self.get("optimization_config")

    def recursion_limit(self) -> int:
        return self.get("recursion_limit")

    def loop_limit(self) -> int:
        return self.get("loop_limit")

    def get_required_keys(self) -> list[str]:
        return ["backend", "mock_config", "optimization_config", "recursion_limit", "loop_limit"]
