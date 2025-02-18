from typing import Any

from zinnia.config.base import ConfigBase


class OptimizationConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.set('always_satisfied_elimination', True)
        self.set('constant_fold', True)
        self.set('dead_code_elimination', True)
        self.set('duplicate_code_elimination', True)
        self.set('shortcut_optimization', True)

    def always_satisfied_elimination(self) -> bool:
        return self.get('always_satisfied_elimination')

    def constant_fold(self) -> bool:
        return self.get('constant_fold')

    def dead_code_elimination(self) -> bool:
        return self.get('dead_code_elimination')

    def duplicate_code_elimination(self) -> bool:
        return self.get('duplicate_code_elimination')

    def shortcut_optimization(self) -> bool:
        return self.get('shortcut_optimization')

    def verify(self, key: str, value: Any) -> Any:
        for _key in ['always_satisfied_elimination', 'constant_fold', 'dead_code_elimination', 'duplicate_code_elimination', 'shortcut_optimization']:
            if key == _key:
                if not isinstance(value, bool):
                    raise ValueError(f'Invalid `{key}` specified: {value}')
                return bool(value)
        return value

    def get_required_keys(self) -> list[str]:
        return ['always_satisfied_elimination', 'constant_fold', 'dead_code_elimination', 'duplicate_code_elimination', 'shortcut_optimization']
