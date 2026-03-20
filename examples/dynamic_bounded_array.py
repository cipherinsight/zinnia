from zinnia import zk_circuit, DynamicNDArray, np


@zk_circuit
def verify_energy_window(
    hourly_kwh: DynamicNDArray[int, 8, 1],
    window_start: int,
    window_end: int,
    expected_total_kwh: int,
    expected_peak_row_kwh: int,
):
    assert 0 <= window_start
    assert window_start <= window_end
    assert window_end <= 8

    # Check metrics for a selected billing window from a bounded dynamic profile.
    billing_window = hourly_kwh[window_start:window_end]
    assert (window_end - window_start) == 4

    window_matrix = billing_window.reshape((2, 2))
    row_totals = window_matrix.sum(axis=1)
    assert row_totals.sum(axis=0) == expected_total_kwh
    assert row_totals.max(axis=0) == expected_peak_row_kwh


assert verify_energy_window([4, 5, 7, 6, 9, 3, 0, 0], 1, 5, 27, 9)
assert verify_energy_window([4, 5, 7, 6, 9, 3, 0, 0], 0, 3, 16, 7)
