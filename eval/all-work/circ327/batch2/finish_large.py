from remaining import *
n='case0387'
SPECS[n]['code']=(BASE/n/'verification.zok').read_text()
SPECS[n]['reason']+=' Static loop expansion uses an array literal to avoid zxi retaining successive full-array assignments; no value-dependent bounds changed.'
execute(n)
