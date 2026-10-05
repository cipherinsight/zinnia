"""Independently authored threshold-energy implementation inventory."""
from pathlib import Path
import json
from textwrap import dedent

CANDIDATES = {
    'mask_product': ('broadcast mask / multiply / sum', 'return ((x > floor) * x * x).sum(axis=-1)'),
    'sum_where': ('NumPy reduction with where and initial', 'return np.sum(x ** 2, axis=1, where=x > floor, initial=0)'),
    'where_square': ('zero inactive readings before squaring', 'active = np.where(x > floor, x, 0)\nreturn np.square(active).sum(axis=1)'),
    'add_reduce': ('ufunc reduction', 'return np.add.reduce(np.where(x > floor, x * x, 0), axis=1)'),
    'einsum_masked': ('einsum of masked readings', "active = np.where(x > floor, x, 0)\nreturn np.einsum('nc,nc->n', active, active)"),
    'row_filtered_dot': ('Boolean selection followed by dot', '''
        result = []
        for row in x:
            active = row[row > floor]
            result.append(np.dot(active, active))
        return np.asarray(result)
    '''),
    'row_filtered_sum': ('Boolean selection followed by sum', 'return np.array([np.sum(row[row > floor] ** 2) for row in x])'),
    'generator_filter': ('generator with filter clause', 'return np.array([sum(v * v for v, f in zip(row, floor) if v > f) for row in x])'),
    'helper_closure': ('local helper captures floor', '''
        def energy(row):
            return sum(v * v for v, f in zip(row, floor) if v > f)
        return np.array([energy(row) for row in x])
    '''),
    'helper_explicit': ('local helper with explicit argument', '''
        def energy(row, limits):
            total = 0
            for v, f in zip(row, limits):
                if v > f:
                    total += v ** 2
            return total
        return np.array([energy(row, floor) for row in x])
    '''),
    'map_lambda': ('map lambda over rows', 'return np.array(list(map(lambda row: np.sum(np.where(row > floor, row * row, 0)), x)))'),
    'filter_lambda': ('filter pairs then map contribution', '''
        result = []
        for row in x:
            pairs = filter(lambda pair: pair[0] > pair[1], zip(row, floor))
            result.append(sum(map(lambda pair: pair[0] * pair[0], pairs)))
        return np.array(result)
    '''),
    'apply_along_axis': ('apply row helper along channel axis', '''
        def energy(row):
            return np.sum(np.where(row > floor, row * row, 0))
        return np.apply_along_axis(energy, 1, x)
    '''),
    'compress_rows': ('compress selects channel values', 'return np.array([np.sum(np.compress(row > floor, row) ** 2) for row in x])'),
    'nonzero_indices': ('nonzero channel index iteration', '''
        result = []
        for row in x:
            total = 0
            for c in np.nonzero(row > floor)[0]:
                total += row[c] ** 2
            result.append(total)
        return np.asarray(result)
    '''),
    'flatnonzero_take': ('flatnonzero and take gather', '''
        result = []
        for row in x:
            active = np.take(row, np.flatnonzero(row > floor))
            result.append(np.sum(active * active))
        return np.array(result)
    '''),
    'masked_copyto': ('copyto inactive positions', '''
        squares = x * x
        np.copyto(squares, 0, where=x <= floor)
        return squares.sum(axis=1)
    '''),
    'boolean_assignment': ('Boolean assignment on copy', '''
        active = x.copy()
        active[x <= floor] = 0
        return (active ** 2).sum(axis=1)
    '''),
    'channel_accumulation': ('column iteration accumulates output vector', '''
        result = np.zeros(x.shape[0], dtype=int)
        for c, f in enumerate(floor):
            values = x[:, c]
            result += np.where(values > f, values ** 2, 0)
        return result
    '''),
    'continue_loop': ('nested loops with continue', '''
        result = []
        for row in x:
            total = 0
            for c, v in enumerate(row):
                if v <= floor[c]:
                    continue
                total += v * v
            result.append(total)
        return np.array(result)
    '''),
    'while_indices': ('static while loops with conditional scalar', '''
        result = np.zeros(x.shape[0], dtype=int)
        n = 0
        while n < x.shape[0]:
            c = 0
            total = 0
            while c < x.shape[1]:
                v = x[n, c]
                total += v * v if v > floor[c] else 0
                c += 1
            result[n] = total
            n += 1
        return result
    '''),
    'list_filtered_append': ('dynamic filtered list append and reduction', '''
        result = []
        for row in x:
            terms = []
            for v, f in zip(row, floor):
                if v > f:
                    terms.append(v * v)
            result.append(sum(terms))
        return np.asarray(result)
    '''),
    'transpose_builtin_sum': ('transpose channels and built-in sum', 'return sum((np.where(column > f, column * column, 0) for column, f in zip(x.T, floor)), np.zeros(x.shape[0], dtype=int))'),
    'masked_matmul': ('masked rows times all-ones vector', 'return np.where(x > floor, x * x, 0) @ np.ones(x.shape[1], dtype=int)'),
    'reduce_keepdims_squeeze': ('sum tuple axis and squeeze', 'return np.sum(np.where(x > floor, x ** 2, 0), axis=(1,), keepdims=True).squeeze(axis=1)'),
}

if __name__ == '__main__':
    root = Path(__file__).parent
    inventory = []
    for name, (approach, body) in CANDIDATES.items():
        source = 'def kernel(x, floor):\n' + ''.join('    ' + line + '\n' for line in dedent(body).strip().splitlines())
        (root / f'{name}.py').write_text(source)
        inventory.append({'name': name, 'approach': approach, 'path': f'{name}.py'})
    (root / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
