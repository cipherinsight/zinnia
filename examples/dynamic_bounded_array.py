from zinnia import zk_circuit, DynamicNDArray


@zk_circuit
def verify_energy_window(
    hourly_kwh: DynamicNDArray[int, 8, 1],
    window_start: int,
    window_end: int,
    expected_total_kwh: int,
    expected_peak_kwh: int,
):
    assert 0 <= window_start
    assert window_start < window_end
    assert window_end <= len(hourly_kwh)

    # Metrics of a billing window whose bounds are private inputs.
    billing_window = hourly_kwh[window_start:window_end]
    assert billing_window.sum() == expected_total_kwh
    assert billing_window.max() == expected_peak_kwh


assert verify_energy_window([4, 5, 7, 6, 9, 3], 1, 5, 27, 9)
assert verify_energy_window([4, 5, 7, 6, 9, 3], 0, 3, 16, 7)
assert not verify_energy_window([4, 5, 7, 6, 9, 3], 0, 3, 17, 7)
