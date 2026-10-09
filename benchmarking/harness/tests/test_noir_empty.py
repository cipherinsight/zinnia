"""A transport adaptation must not hide constraints or public inputs."""
import base64
import gzip
import json

import msgpack
import pytest

from benchmarking.noir_witness import for_barretenberg


@pytest.mark.parametrize('private_count', [0, 1, 48, 100])
def test_constraint_free_witness_has_one_unused_slot(tmp_path, private_count):
    program = [[['main', max(0, private_count-1), [], list(range(private_count)), [], [], []]], []]
    artifact = tmp_path / 'program.json'
    artifact.write_text(json.dumps({'bytecode': base64.b64encode(gzip.compress(
        b'\x03' + msgpack.packb(program))).decode()}))
    witness = tmp_path / 'witness.gz'
    original = gzip.compress(b'original executor witness')
    witness.write_bytes(original)
    adapted = for_barretenberg(artifact, witness)
    raw = gzip.decompress(adapted.read_bytes())
    assert raw[0] == 3
    assert msgpack.unpackb(raw[1:], strict_map_key=False) == [[[0, {0: bytes(32)}]]]
    assert witness.read_bytes() == original


@pytest.mark.parametrize('field,value', [(2, ['opcode']), (4, [0]), (5, [0])])
def test_nonempty_or_public_circuit_is_untouched(tmp_path, field, value):
    circuit = ['main', 0, [], [0], [], [], []]
    circuit[field] = value
    artifact = tmp_path / 'program.json'
    artifact.write_text(json.dumps({'bytecode': base64.b64encode(gzip.compress(
        b'\x03' + msgpack.packb([[circuit], []]))).decode()}))
    witness = tmp_path / 'witness.gz'
    witness.write_bytes(b'unchanged')
    assert for_barretenberg(artifact, witness) == witness
    assert witness.read_bytes() == b'unchanged'
