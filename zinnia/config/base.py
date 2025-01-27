from typing import Any


class ConfigBase:
    def __init__(self):
        self.__config = {}

    def get(self, key: str) -> Any:
        return self.__config.get(key, None)

    def set(self, key: str, value: Any):
        self.__config[key] = self.verify(key, value)

    def serialize(self) -> dict:
        the_result = {}
        for key, value in self.__config.items():
            if isinstance(value, ConfigBase):
                the_result[key] = value.serialize()
            else:
                the_result[key] = value
        return the_result

    def deserialize(self, config: dict) -> 'ConfigBase':
        self.__config = {}
        for key, value in config.items():
            self.set(key, self.verify(key, value))
        required_keys = self.get_required_keys()
        for key in required_keys:
            if key not in self.__config:
                raise ValueError(f"Required key {key} not found in config")
        return self

    def verify(self, key: str, value: Any) -> Any:
        return value

    def get_required_keys(self) -> list[str]:
        return []
