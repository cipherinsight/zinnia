from refinements import *
from finalize import finalize
for name in ENTRIES:
    execute(name)
finalize()
