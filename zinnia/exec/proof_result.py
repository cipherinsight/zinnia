"""Wrapper around a proof artifact produced by the Rust proving backend."""

import json


class ZKProofResult:
    """A self-contained proof artifact.

    Contains the serialized verification key, proof bytes, and public
    inputs so that verification requires only this object.
    """

    def __init__(self, backend, vk_bytes_hex, proof_bytes_hex, public_values, k):
        self.backend = backend
        self.vk_bytes_hex = vk_bytes_hex
        self.proof_bytes_hex = proof_bytes_hex
        self.public_values = public_values
        self.k = k

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "backend": self.backend,
            "vk_bytes": self.vk_bytes_hex,
            "proof_bytes": self.proof_bytes_hex,
            "public_values": self.public_values,
            "k": self.k,
        })

    @staticmethod
    def from_json(json_str: str) -> 'ZKProofResult':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return ZKProofResult(
            backend=data["backend"],
            vk_bytes_hex=data["vk_bytes"],
            proof_bytes_hex=data["proof_bytes"],
            public_values=data["public_values"],
            k=data["k"],
        )

    def save(self, path: str):
        """Save proof artifact to a file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> 'ZKProofResult':
        """Load proof artifact from a file."""
        with open(path, 'r') as f:
            return ZKProofResult.from_json(f.read())

    def __repr__(self):
        proof_len = len(self.proof_bytes_hex) // 2
        return f"ZKProofResult(backend={self.backend!r}, k={self.k}, proof_size={proof_len}B)"
