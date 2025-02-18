from typing import Tuple, List, Any


class ZKParsedInput:
    class Kind:
        FLOAT = "Float"
        INTEGER = "Integer"
        HASH = "Hash"

    class Entry:
        def __init__(self, indices: Tuple[int, ...], kind: str, value: Any):
            self.indices = indices
            self.kind = kind
            self.value = value

        def get_indices(self) -> Tuple[int, ...]:
            return self.indices

        def get_kind(self) -> str:
            return self.kind

        def get_key(self) -> str:
            if self.kind == ZKParsedInput.Kind.FLOAT:
                return f"x_{'_'.join(map(str, self.indices))}"
            if self.kind == ZKParsedInput.Kind.INTEGER:
                return f"x_{'_'.join(map(str, self.indices))}"
            if self.kind == ZKParsedInput.Kind.HASH:
                return f"hash_{'_'.join(map(str, self.indices))}"
            raise NotImplementedError()

        def is_float(self) -> bool:
            return self.kind == ZKParsedInput.Kind.FLOAT

        def is_integer(self) -> bool:
            return self.kind == ZKParsedInput.Kind.INTEGER

        def is_hash(self) -> bool:
            return self.kind == ZKParsedInput.Kind.HASH

        def get_value(self) -> Any:
            return self.value

        def __str__(self):
            return f'Entry(indices={self.indices}, kind="{self.kind}", value={self.value})'

    def __init__(self, entries: List['ZKParsedInput.Entry']):
        self.entries = entries

    def get_entries(self) -> List['ZKParsedInput.Entry']:
        return self.entries

    def __str__(self):
        return f"ZKParsedInput(entries={', '.join(map(str, self.entries))})"
