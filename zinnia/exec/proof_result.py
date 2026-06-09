"""Wrapper around a proof artifact produced by the Rust proving backend."""

import json


class ZKProofResult:
    """A self-contained proof artifact.

    Contains the serialized verification key, proof bytes, and public
    inputs so that verification requires only this object.
    """

    def __init__(
        self,
        backend,
        vk_bytes_hex,
        proof_bytes_hex,
        public_values,
        k,
        format="",
        ir_hash="",
        prove_seconds=0.0,
    ):
        self.backend = backend
        self.format = format
        self.vk_bytes_hex = vk_bytes_hex
        self.ir_hash = ir_hash
        self.proof_bytes_hex = proof_bytes_hex
        self.public_values = public_values
        self.k = k
        # Wall-clock duration of the halo2 `create_proof` call alone, in
        # seconds. Matches what halo2-graph's scaffold measures (the
        # "Proving time: ..." line). Use for apples-to-apples benchmarks.
        self.prove_seconds = prove_seconds

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "backend": self.backend,
            "format": self.format,
            "vk_bytes": self.vk_bytes_hex,
            "ir_hash": self.ir_hash,
            "proof_bytes": self.proof_bytes_hex,
            "public_values": self.public_values,
            "k": self.k,
            "prove_seconds": self.prove_seconds,
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
            format=data.get("format", ""),
            ir_hash=data.get("ir_hash", ""),
            prove_seconds=data.get("prove_seconds", 0.0),
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
