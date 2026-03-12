from zinnia.compile.type_sys import DTDescriptorFactory, IntegerType
from zinnia.compile.type_sys.dynamic_ndarray import DynamicNDArrayDTDescriptor


def test_dynamic_ndarray_descriptor_create_from_factory():
    dt = DTDescriptorFactory.create(None, "DynamicNDArray", (IntegerType, 128, 4))
    assert isinstance(dt, DynamicNDArrayDTDescriptor)
    assert dt.dtype == IntegerType
    assert dt.max_length == 128
    assert dt.max_rank == 4
    # Dynamic arrays are represented as bounded flat storage in the current type layer.
    assert dt.shape == (128,)


def test_dynamic_ndarray_descriptor_roundtrip_export_import():
    dt = DynamicNDArrayDTDescriptor(IntegerType, 256, 6)
    payload = DTDescriptorFactory.export(dt)
    restored = DTDescriptorFactory.import_from(payload)

    assert isinstance(restored, DynamicNDArrayDTDescriptor)
    assert restored == dt
