kernel.py:10-12: crop/where lists are dynamic but at most 4x4; masks and padding can bound them. Lines 21-26 call log on private crop entries; lines 28-29 additionally need fixed-point division.
