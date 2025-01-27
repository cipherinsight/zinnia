from typing import Any

from zinnia.config.base import ConfigBase
from zinnia.config.mock_exec_config import MockExecConfig


class ZinniaConfig(ConfigBase):
    BACKEND_HALO2 = "halo2"

    def __init__(self):
        super().__init__()
        self.set("backend", self.BACKEND_HALO2)
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
        return value

    def get_backend(self) -> str:
        return self.get("backend")

    def mock_config(self):
        return self.get("mock_config")

    def get_required_keys(self) -> list[str]:
        return ["backend", "mock_config"]
