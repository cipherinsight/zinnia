from typing import Any

from zinnia.config.base import ConfigBase


class MockExecConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.set('float_tolerance', 1e-6)

    def float_tolerance(self) -> float:
        return self.get('float_tolerance')

    def verify(self, key: str, value: Any) -> Any:
        if key == 'float_tolerance':
            if not isinstance(value, float) and not isinstance(value, int):
                raise ValueError(f'Invalid `float_tolerance` specified: {value}')
            return float(value)
        return value

    def get_required_keys(self) -> list[str]:
        return ['float_tolerance']
