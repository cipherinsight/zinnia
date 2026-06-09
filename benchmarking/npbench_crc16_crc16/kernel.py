from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def crc16(data: Float, poly: Float=33800):
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
