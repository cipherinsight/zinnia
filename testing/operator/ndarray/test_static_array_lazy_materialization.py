from zinnia.api.zk_circuit import ZKCircuit


def test_static_array_dynamic_read_materializes_lazy_payload():
    source = """
def lazy_dynamic_read(i: int):
    a = np.asarray([1, 2, 3])
    assert a[i] == 2
"""
    circuit = ZKCircuit.from_source("lazy_dynamic_read", source)
    assert circuit.prove(1, backend="mock").proof_bytes_hex == "mock_satisfied"
    assert circuit.prove(0, backend="mock").proof_bytes_hex != "mock_satisfied"


def test_static_array_dynamic_slice_materializes_lazy_payload():
    source = """
def lazy_dynamic_slice(i: int):
    a = np.asarray([1, 2, 3])
    b = a[i:i+1]
    assert b[0] == 2
"""
    circuit = ZKCircuit.from_source("lazy_dynamic_slice", source)
    assert circuit.prove(1, backend="mock").proof_bytes_hex == "mock_satisfied"


def test_static_array_dynamic_row_materializes_lazy_payload():
    source = """
def lazy_dynamic_row(i: int):
    a = np.asarray([[1, 2], [3, 4]])
    r = a[i]
    assert r[0] == 3
"""
    circuit = ZKCircuit.from_source("lazy_dynamic_row", source)
    assert circuit.prove(1, backend="mock").proof_bytes_hex == "mock_satisfied"


def test_static_array_dynamic_axis_slice_materializes_lazy_payload():
    source = """
def lazy_dynamic_axis_slice(i: int):
    a = np.asarray([[1, 2, 3], [4, 5, 6]])
    b = a[:, i:i+1]
    assert b[0][0] == 2
"""
    circuit = ZKCircuit.from_source("lazy_dynamic_axis_slice", source)
    assert circuit.prove(1, backend="mock").proof_bytes_hex == "mock_satisfied"
