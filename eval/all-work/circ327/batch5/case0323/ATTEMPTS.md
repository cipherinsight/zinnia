# case0323

Category: `needs_gadget: sin`

No native/stdlib Q32 sine; arbitrary signed degrees cannot be substituted by fixture constants.

as_written: needs gadget

kernel.py:11: result = np.sin(np.deg2rad(degree)). No native/stdlib Q32 sine; arbitrary signed degrees cannot be substituted by fixture constants.

rewrite: needs gadget

No native/stdlib Q32 sine; arbitrary signed degrees cannot be substituted by fixture constants.

