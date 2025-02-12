from typing import Any

from zinnia.config.base import ConfigBase
from zinnia.config.mock_exec_config import MockExecConfig


class ZinniaConfig(ConfigBase):
    BACKEND_HALO2 = "halo2"
    DEFAULT_RECURSION_LIMIT = 100
    DEFAULT_LOOP_LIMIT = 1000

    def __init__(self):
        super().__init__()
        self.set("backend", self.BACKEND_HALO2)
        self.set("recursion_limit", self.DEFAULT_RECURSION_LIMIT)
        self.set("loop_limit", self.DEFAULT_LOOP_LIMIT)
        self.set("mock_config", MockExecConfig())

    def verify(self, key: str, value: Any) -> Any:
        if key == "backend":
            if value not in [self.BACKEND_HALO2]:
                raise ValueError(f"Invalid `backend` specified: {value}")
            return value
        elif key == "mock_config":
            if isinstance(value, MockExecConfig):
                return value
            elif isinstance(value, dict):
                return MockExecConfig().deserialize(value)
            raise ValueError(f"Invalid `mock_config` specified: {value}")
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

    def mock_config(self):
        return self.get("mock_config")

    def recursion_limit(self):
        return self.get("recursion_limit")

    def loop_limit(self):
        return self.get("loop_limit")

    def get_required_keys(self) -> list[str]:
        return ["backend", "mock_config", "recursion_limit", "loop_limit"]
