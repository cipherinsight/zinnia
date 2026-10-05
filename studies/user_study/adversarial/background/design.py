"""Generate retained submissions; each file contains only the site kernel."""
from pathlib import Path
import textwrap,json
root=Path(__file__).parent
programs={
'axis_tuple': 'return x - np.min(x, axis=(0,))',
'negative_axis': 'return x - x.min(axis=-2)',
'ufunc_reduce': 'return x - np.minimum.reduce(x, axis=0)',
'ufunc_accumulate': 'return x - np.minimum.accumulate(x, axis=0)[-1]',
'row_minimum': '''low = x[0].copy()
for row in x[1:]:
    low = np.minimum(low, row)
return x - low''',
'where_reduce': '''low = x[0].copy()
for row in x[1:]:
    low = np.where(row < low, row, low)
return np.subtract(x, low)''',
'enumerated_columns': '''out = np.empty_like(x)
for c, column in enumerate(x.T):
    out[:, c] = column - min(column)
return out''',
'zip_columns': '''columns = list(zip(*x))
low = np.array([min(column) for column in columns])
return x - low''',
'generator_min': '''low = np.array([min(x[t, c] for t in range(x.shape[0])) for c in range(x.shape[1])])
return x - low''',
'list_append': '''rows = []
low = x.min(axis=0)
for row in x:
    rows.append(row - low)
return np.array(rows)''',
'column_stack': 'return np.column_stack([column - column.min() for column in x.T])',
'stack_axis': 'return np.stack([column - min(column) for column in x.T], axis=1)',
'nested_helper': '''def correct(column):
    return column - min(column)
return np.array([correct(column) for column in x.T]).T''',
'helper_closure': '''def value(t, c):
    return x[t, c] - min(x[:, c])
return np.array([[value(t, c) for c in range(x.shape[1])] for t in range(x.shape[0])])''',
'copy_inplace': '''out = x.copy()
out -= x.min(axis=0)
return out''',
'subtract_out': '''out = np.empty_like(x)
np.subtract(x, x.min(axis=0), out=out)
return out''',
'min_initial': 'return x - np.min(x, axis=0, initial=4096)',
'min_out': '''low = np.empty(x.shape[1], dtype=x.dtype)
np.min(x, axis=0, out=low)
return x - low''',
'sorted_columns': 'return x - np.sort(x, axis=0)[0]',
'argmin_take': '''idx = np.argmin(x, axis=0)
low = x[idx, np.arange(x.shape[1])]
return x - low''',
'while_loops': '''out = np.zeros_like(x)
c = 0
while c < x.shape[1]:
    low = x[0, c]
    t = 1
    while t < x.shape[0]:
        low = min(low, x[t, c])
        t += 1
    t = 0
    while t < x.shape[0]:
        out[t, c] = x[t, c] - low
        t += 1
    c += 1
return out''',
'sentinel_loop': '''out = np.zeros_like(x)
for c in range(x.shape[1]):
    low = 4096
    for item in x[:, c]:
        low = min(low, item)
    out[:, c] = x[:, c] - low
return out''',
'masked_min': 'return x - np.min(x, axis=0, where=np.ones_like(x, dtype=bool), initial=4096)',
'list_conversion': '''rows = x.tolist()
low = [min(row[c] for row in rows) for c in range(x.shape[1])]
return [[row[c] - low[c] for c in range(x.shape[1])] for row in rows]''',
}
for name,body in programs.items():
    (root/(name+'.py')).write_text('def kernel(x):\n'+textwrap.indent(body,'    ')+'\n')
(root/'inventory.json').write_text(json.dumps([{'name':n,'path':n+'.py','intent':'valid ordinary Python/NumPy background subtraction'} for n in programs],indent=2)+'\n')
