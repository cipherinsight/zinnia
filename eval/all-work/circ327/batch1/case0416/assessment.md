# case0416

Source: kernel.py:11

The dynamic reshape can be padded to 8x6 (six columns bound the number of bins), but mean(axis=-1) then needs signed fixed-point division by bin_size.

As-written: not expressible.
Minimal rewrite: needs_gadget: division.
