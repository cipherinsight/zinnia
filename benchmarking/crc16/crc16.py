# Source: NPBench crc16 (crc16_numpy.py)
# Original signature: crc16(data, poly=0x8408) — data is a length-N uint8 array.
# Migration notes:
#   - N picked from the "S" preset (1600).
#   - poly hoisted to a module-level constant (default arg).
from zinnia import *

N = 1600
POLY = 0x8408


@zk_circuit
def crc16(data: NDArray[Integer, 1600]):
    '''
    CRC-16-CCITT Algorithm
    '''
    crc = 0xFFFF
    for b in data:
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ POLY
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = (~crc & 0xFFFF)
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    _zinnia_result = crc & 0xFFFF
