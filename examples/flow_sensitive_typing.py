from zinnia import zk_circuit, DynamicNDArray, NDArray, np


@zk_circuit
def reserve_from_forecast(
    nowcast_kw: DynamicNDArray[int, 8, 1],
    climatology_kw: DynamicNDArray[int, 4, 1],
    nowcast_is_reliable: int,
    committed_supply_kw: int
):
    if nowcast_is_reliable:
        source = nowcast_kw
        effective_len = 8
    else:
        source = climatology_kw
        effective_len = 4

    active_window = source[:effective_len]
    peak_kw = active_window.max(axis=0)
    reserve_kw = peak_kw - committed_supply_kw
    return reserve_kw
