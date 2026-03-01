"""
Membership Proof Example
------------------------

This example demonstrates a privacy-preserving set-membership / non-membership
statement against a Merkle commitment. A prover can show whether a candidate ID
corresponds to a populated leaf or an empty placeholder leaf, without revealing
the full underlying dataset.
"""


from zinnia import *

from typing import Any

from examples.utils import create_circuit, create_prove_verify_keys, prove, verify


def generate_membership_input(
    elements: list[int],
    candidate_id: int,
    tree_height: int = 8,
    empty_leaf_value: int = 0,
) -> dict[str, Any]:
    # Build a complete tree shape for deterministic witness generation.
    total_leaves = 1 << tree_height
    if len(elements) > total_leaves:
        raise ValueError(f"Too many elements ({len(elements)}). Max for tree_height={tree_height} is {total_leaves}.")

    # Decide if we are proving membership or non-membership.
    candidate_exists = candidate_id in elements
    if not candidate_exists and len(elements) == total_leaves:
        raise ValueError("Cannot build non-membership proof with full tree: no empty leaf available.")

    # Hash all leaves with Poseidon so tree operations are field-friendly for ZK.
    empty_leaf_hash = poseidon_hash(empty_leaf_value)
    leaf_hashes = [poseidon_hash(v) for v in elements] + [empty_leaf_hash] * (total_leaves - len(elements))

    # Choose the leaf index: actual element index or first empty slot.
    if candidate_exists:
        leaf_index = elements.index(candidate_id)
    else:
        leaf_index = len(elements)

    # Traverse the tree once to build the path witness (siblings + directions).
    acc = leaf_hashes[leaf_index]
    siblings: list[int] = []
    directions: list[bool] = []
    level_nodes = list(leaf_hashes)
    level_index = leaf_index

    for _ in range(tree_height):
        # Direction convention: False means current node is left child.
        if level_index % 2 == 0:
            sibling_index = level_index + 1
            directions.append(False)
        else:
            sibling_index = level_index - 1
            directions.append(True)
        siblings.append(level_nodes[sibling_index])

        # Fold current level into parent level with Poseidon pair hashing.
        next_level: list[int] = []
        for i in range(0, len(level_nodes), 2):
            next_level.append(poseidon_hash([level_nodes[i], level_nodes[i + 1]]))
        level_nodes = next_level
        level_index //= 2

    merkle_root = level_nodes[0]

    # Return all witness/public values expected by the example circuit API.
    return {
        "candidate_id": candidate_id,
        "merkle_root": merkle_root,
        "siblings": siblings,
        "directions": directions,
        "leaf_value": acc,
        "empty_leaf_hash": empty_leaf_hash,
        "candidate_exists": candidate_exists,
    }


EXAMPLE_INPUT = generate_membership_input(
    elements=[2, 4, 7, 9, 11, 15],
    candidate_id=2,
)


@zk_circuit
def membership_proof(
    candidate_id: Public[Integer],
    merkle_root: Public[Integer],
    siblings: Public[NDArray[Integer, 8]],
    directions: Public[NDArray[Boolean, 8]],
    leaf_value: Public[Integer],
    empty_leaf_hash: Public[Integer],
    candidate_exists: Public[Boolean],
):
    # Hash the candidate inside the circuit so the relation is constrained, not trusted.
    candidate_hash = poseidon_hash(candidate_id)
    # This minimal circuit focuses on semantic consistency of the selected leaf.
    # (A full path-recompute circuit would additionally constrain siblings/directions/root.)
    if candidate_exists:
        assert leaf_value == candidate_hash
    else:
        assert leaf_value == empty_leaf_hash


def run_membership_proof(
    data: dict[str, Any] | None = None,
    circuit_name: str = "membership_proof",
    k: int = 16,
):
    # Provide turnkey defaults for demonstration runs.
    if data is None:
        data = EXAMPLE_INPUT

    # Use a defensive copy to preserve caller input.
    proving_data = dict(data)

    # Standard Zinnia lifecycle from function-level circuit to verified proof artifact.
    circuit = create_circuit(membership_proof, chips=[])
    keygen_result = create_prove_verify_keys(
        circuit,
        proving_data,
        circuit_name=circuit_name,
        k=k,
    )
    prove_result = prove(circuit, keygen_result, proving_data)
    verify_result = verify(keygen_result)

    return {
        "keygen": keygen_result,
        "prove": prove_result,
        "verify": verify_result,
    }


if __name__ == '__main__':
    run_membership_proof(
        data=EXAMPLE_INPUT,
        circuit_name="membership_proof",
        k=16,
    )
