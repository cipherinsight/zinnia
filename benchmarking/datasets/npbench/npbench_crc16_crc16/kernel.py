# npbench program crc16/crc16_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def crc16(data: Integer[Array, '16'], poly: Integer=33800):
    """
    CRC-16-CCITT Algorithm
    """
    crc = 65535
    for b in data:
        cur_byte = 255 & b
        for _ in range(0, 8):
            if crc & 1 ^ cur_byte & 1:
                crc = crc >> 1 ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = ~crc & 65535
    crc = crc << 8 | crc >> 8 & 255
    return crc & 65535
