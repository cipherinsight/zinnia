"""Study contracts and server-owned correctness oracles."""
import copy
import numpy as np
from ..tasks import TASKS, fixtures, oracle

ORDER = [t.name for t in TASKS]
BUDGETS = dict(zip(ORDER, [300, 420, 540, 720, 720, 900]))
TITLES = {
    "practice": ["Practice: add two arrays", "练习：数组相加"],
    "calibrate": ["Calibrate sensor readings", "校准传感器读数"],
    "background": ["Remove channel background", "去除通道背景值"],
    "threshold_energy": ["Sum squared readings above a threshold", "对超过阈值的读数求平方和"],
    "distances": ["Pairwise squared distances", "计算两两距离的平方"],
    "pool": ["Sum image blocks", "图像分块求和"],
    "smooth": ["Update a grid over three rounds", "对网格进行三轮更新"],
}
DESCRIPTIONS = {'practice': ['You are given two one-dimensional integer arrays, x and y, of the same length. Add the values '
              'at each matching position and return the sums in a new array of that length.\n'
              '\n'
              'For example, x = [1, 2, 3] and y = [4, 5, 6] should produce [5, 7, 9].\n'
              '\n'
              'This practice is untimed and unscored. Complete the starter code, then use Run examples to '
              'check your result.',
              '输入是两个长度相同的一维整数数组 x 和 y。请将两个数组中相同位置的值相加，并把各位置的和组成一个同样长度的新数组返回。\n'
              '\n'
              '例如，x = [1, 2, 3]、y = [4, 5, 6] 时，应返回 [5, 7, 9]。\n'
              '\n'
              '本练习没有时间限制，也不计分。请补全初始代码，再点击“运行示例”检查结果。'],
 'calibrate': ['The one-dimensional array x contains N sensor readings. For each reading, first multiply it '
               'by 2, then add 5. If the result is below 0, replace it with 0; if it is above 255, replace '
               'it with 255. Otherwise, keep the result as it is.\n'
               '\n'
               'Return N adjusted readings in the same order as the input.\n'
               '\n'
               'For example, x = [-10, 0, 50, 200] should produce [0, 5, 105, 255].',
               '一维数组 x 中有 N 个传感器读数。对每个读数，先乘以 2，再加 5。如果计算结果小于 0，就改为 0；如果大于 255，就改为 255；其余结果保持不变。\n'
               '\n'
               '请按输入的原顺序，返回包含 N 个调整后读数的数组。\n'
               '\n'
               '例如，x = [-10, 0, 50, 200] 时，应返回 [0, 5, 105, 255]。'],
 'background': ['The matrix x has T rows and C columns. Each row is one measurement, and each column is a '
                'sensor channel. For this task, the background value of a channel is the smallest reading in '
                'its column.\n'
                '\n'
                'Find that value separately for each column, then subtract it from every reading in the same '
                'column. Return a matrix with the same rows and columns as x. If there is only one row, the '
                'result is all zeros.\n'
                '\n'
                'For example, x = [[3, 10], [5, 7], [4, 9]] has column minima 3 and 7. Subtracting them '
                'gives [[0, 3], [2, 0], [1, 2]].',
                '矩阵 x 有 T 行、C 列。每一行是一次测量，每一列是一个传感器通道。在本任务中，一个通道的“背景值”就是该列读数中的最小值。\n'
                '\n'
                '请分别找出每列的背景值，再用该列的每个读数减去它。返回矩阵的行数和列数应与 x 相同。如果输入只有一行，所有结果都应为零。\n'
                '\n'
                '例如，x = [[3, 10], [5, 7], [4, 9]] 的两列最小值分别是 3 和 7。各列减去对应值后，结果为 [[0, 3], [2, 0], [1, 2]]。'],
 'threshold_energy': ['The matrix x has N rows of readings from C channels. The array floor contains C '
                      'thresholds: floor[c] is the threshold for column c.\n'
                      '\n'
                      'For each row, keep only readings that are greater than their own channel threshold. '
                      'Square those readings and add the squares. Return one total per row, in an array of '
                      'length N. A reading equal to its threshold is excluded; a row with no qualifying '
                      'readings has total 0. Readings and thresholds can be negative.\n'
                      '\n'
                      'For example, with row [2, 5, 1] and thresholds [0, 4, 3], the qualifying readings are '
                      '2 and 5. The total is 2² + 5² = 29.',
                      '矩阵 x 有 N 行读数，每行包含 C 个通道的值。数组 floor 中有 C 个阈值，其中 floor[c] 对应第 c 列。\n'
                      '\n'
                      '对每一行，只保留大于各自通道阈值的读数，将这些读数分别平方，再把平方值相加。请返回长度为 N '
                      '的数组，每个位置是对应行的总和。等于阈值的读数不计入；如果一行没有符合条件的读数，该行结果为 0。读数和阈值都可能为负数。\n'
                      '\n'
                      '例如，一行读数为 [2, 5, 1]，阈值为 [0, 4, 3] 时，符合条件的是 2 和 5，因此结果为 2² + 5² = 29。'],
 'distances': ['The matrices x and y contain N and M points respectively. Each row is one point, described '
               'by D coordinates.\n'
               '\n'
               'Return an N-row, M-column matrix where result[i, j] is the squared distance from row i of x '
               'to row j of y. To calculate it, subtract matching coordinates, square each difference, and '
               'add the squares. Do not take a square root. N and M may differ, and points may repeat.\n'
               '\n'
               'For example, the squared distance from [1, 2] to [3, 2] is (1 − 3)² + (2 − 2)² = 4.',
               'x 的每一行表示一个点，行内的 D 个值是该点的坐标；x 一共有 N 个点。y 用同样的方式表示 M 个点，每个点也有 D 个坐标。\n'
               '\n'
               '请计算 x 中每个点到 y 中每个点的距离平方。对于一对点，先将对应坐标相减，把差分别平方，再相加。返回一个 N 行、M 列的矩阵，其中 result[i, j] 对应 x 的第 i '
               '行与 y 的第 j 行。结果不需要开平方。两组点的数量可以不同，也可以出现重复的点。\n'
               '\n'
               '例如，[1, 2] 到 [3, 2] 的距离平方为 (1 − 3)² + (2 − 2)² = 4。'],
 'pool': ['The array x has shape [B, H, W, C]: B images, each with H rows, W columns, and C channels per '
          'pixel. H and W are positive even numbers.\n'
          '\n'
          'Starting at the top-left corner, divide each image into 2 × 2 blocks so each pixel belongs to one '
          'block. Within each block, sum the four values for each channel separately. Preserve the block '
          'order, keeping images and channels separate. Return an array with shape [B, H/2, W/2, C].\n'
          '\n'
          'For one image and channel, rows [1, 2, 3, 4] and [5, 6, 7, 8] produce one row [14, 22]: the left '
          'block sums to 14, and the right to 22.',
          '数组 x 的形状为 [B, H, W, C]，表示 B 张图像。每张图像有 H 行、W 列，每个像素有 C 个通道值。H 和 W 都是正偶数。\n'
          '\n'
          '请从每张图像的左上角开始，将相邻的两行、两列划成一个 2 × 2 '
          '区块，每个像素只属于一个区块。在每个区块内，分别将同一通道的四个值相加。不同图像、不同通道之间不要混合，区块的空间顺序保持不变。返回数组的形状应为 [B, H/2, W/2, C]。\n'
          '\n'
          '例如，单张、单通道图像的两行为 [1, 2, 3, 4] 和 [5, 6, 7, 8]。两个区块的和分别为 1 + 2 + 5 + 6 = 14 和 3 + 4 + 7 + 8 = '
          '22，因此输出只有一行 [14, 22]。'],
 'smooth': ['The matrix x is an H-row, W-column grid of nonnegative integers. Update it three times using '
            'the following rule.\n'
            '\n'
            'Keep the outermost rows and columns unchanged. For each interior cell, multiply its value by 4, '
            'add its four neighbours (above, below, left, right), then divide by 8 and round down.\n'
            '\n'
            'Use only values from the start of the round: newly calculated values must not affect other '
            'cells until the next round. Return the full grid after three rounds, with shape [H, W]. Grids '
            'with fewer than three rows or columns stay unchanged.\n'
            '\n'
            'For example, in a 3 × 3 grid with centre 8 and all other values 0, the centre becomes 4, then '
            '2, then 1. The surrounding values stay 0.',
            '矩阵 x 是一个 H 行、W 列的网格，里面都是非负整数。请按照同一规则，连续更新三轮。\n'
            '\n'
            '每一轮都保持最外侧的行和列不变。对于其余单元，将自身的值乘以 4，加上正上、正下、正左、正右四个相邻单元的值，再将总和除以 8，向下取整。\n'
            '\n'
            '同一轮的所有计算都必须使用这一轮开始时的值。本轮已经算出的新值，要到下一轮才能用于其他单元的计算。完成三轮后，返回形状仍为 [H, W] 的完整网格。如果行数或列数小于 '
            '3，就没有需要更新的内部单元。\n'
            '\n'
            '例如，一个 3 × 3 网格的中心为 8，其余值都为 0。三轮更新后，中心依次变为 4、2、1，周围的值始终为 0。']}
HINTS = {'practice': (['Combine values at matching positions. Elementwise array addition or a loop over indices both '
               'work. A straightforward implementation using only loops (nested where needed) is also '
               'accepted.',
               '将相同位置的值相加。可以使用数组逐元素加法，也可以遍历索引。 也可以采用最直接的循环实现，逐个计算；需要时使用嵌套循环。'],
              ['shape', 'broadcast']),
 'calibrate': (['Apply the scale and offset to each element first, then clamp each result between the lower '
                'and upper bounds. Preserve the original shape. A straightforward implementation using only '
                'loops (nested where needed) is also accepted.',
                '先对每个元素进行缩放和偏移，再将结果限制在上下界之间。保留原数组的形状。 也可以采用最直接的循环实现，逐个计算；需要时使用嵌套循环。'],
               ['clamp', 'broadcast']),
 'background': (['First reduce the sample axis (axis 0) with minimum to obtain one value per channel. Then '
                 'subtract that channel vector from each row, using broadcasting or nested loops. A '
                 'straightforward implementation using only loops (nested where needed) is also accepted.',
                 '先沿采样轴（轴 0）取最小值，得到每个通道的背景值。再通过广播或嵌套循环，将每行减去该通道向量。 也可以采用最直接的循环实现，逐个计算；需要时使用嵌套循环。'],
                ['minmax', 'broadcast']),
 'threshold_energy': (['Compare each reading with its channel threshold before squaring. Replace excluded '
                       'contributions with zero, then sum along the channel axis (axis 1). Use a strict '
                       'greater-than comparison. A straightforward implementation using only loops (nested '
                       'where needed) is also accepted.',
                       '先将读数与对应通道的阈值比较，再对符合条件的读数求平方。不符合条件的贡献置零，最后沿通道轴（轴 1）求和。注意使用严格大于比较。 '
                       '也可以采用最直接的循环实现，逐个计算；需要时使用嵌套循环。'],
                      ['aggregate', 'sum', 'broadcast']),
 'distances': (['For each pair of rows, subtract matching coordinates, square the differences, and reduce '
                'the coordinate axis by summation. You can loop over row pairs, or form a broadcasted '
                '[N,M,D] difference array and reduce its last axis. A straightforward implementation using '
                'only loops (nested where needed) is also accepted.',
                '对每一对行，将对应坐标相减、求平方，再沿坐标轴求和。可以遍历行对，也可以通过广播构造 [N,M,D] 的差值数组，再沿最后一个轴归约。 '
                '也可以采用最直接的循环实现，逐个计算；需要时使用嵌套循环。'],
               ['sum', 'expand', 'shape']),
 'pool': (['Keep batch and channel indices separate. Combine the four spatial offsets (even/even, even/odd, '
           'odd/even, odd/odd) using stepped slices or loops. Alternatively reshape into [B,H/2,2,W/2,2,C] '
           'and sum the two within-block axes; account for axis renumbering after each reduction. A '
           'straightforward implementation using only loops (nested where needed) is also accepted.',
           '保持批次和通道索引不变。通过步长切片或循环，组合四个空间偏移（偶/偶、偶/奇、奇/偶、奇/奇）。也可以重塑为 '
           '[B,H/2,2,W/2,2,C]，再对两个块内轴求和；注意每次归约后的轴编号变化。 也可以采用最直接的循环实现，逐个计算；需要时使用嵌套循环。'],
          ['slicing', 'reshape', 'sum']),
 'smooth': (['Use separate current and next arrays for every round. Start next as a copy to preserve the '
             'boundary, update only interior cells from current, then replace current after the whole round. '
             'Handle grids without an interior by leaving them unchanged. A straightforward implementation '
             'using only loops (nested where needed) is also accepted.',
             '每轮使用独立的当前数组和下一轮数组。先复制当前数组以保留边界，只根据当前数组更新内部单元，整轮结束后再替换当前数组。没有内部单元的网格应保持不变。 '
             '也可以采用最直接的循环实现，逐个计算；需要时使用嵌套循环。'],
            ['updates', 'slicing', 'allocate'])}

RANKS = {"practice": [1, 1], "calibrate": [1], "background": [2], "threshold_energy": [2, 1], "distances": [2, 2], "pool": [4], "smooth": [2]}
OUT_RANK = {"practice": 1, "calibrate": 1, "background": 2, "threshold_energy": 1, "distances": 2, "pool": 4, "smooth": 2}


def names(task):
    return ["x", "floor"] if task == "threshold_energy" else (["x", "y"] if task in ("practice", "distances") else ["x"])


def starter(task, condition):
    if condition == "zinnia":
        config = "    R = 3\n" if task == "smooth" else ""
        return f"def kernel({', '.join(names(task))}):\n{config}    # Write your implementation here.\n    raise NotImplementedError(\"Implement kernel\")\n"
    params = ", ".join(f"{n}: &Array{rank}<i64>" for n, rank in zip(names(task), RANKS[task]))
    config = "    let rounds = 3;\n" if task == "smooth" else ""
    return f"fn kernel({params}) -> Array{OUT_RANK[task]}<i64> {{\n{config}    // Write your implementation here.\n    todo!(\"Implement kernel\")\n}}\n"


def cases(task, hidden=False):
    if task == "practice":
        return [{"x": [1, 2, 3], "y": [4, 5, 6]}, {"x": [0], "y": [0]}, {"x": [-2, 5], "y": [2, -7]}]
    contract = next(t for t in TASKS if t.name == task)
    selections = [(0, i) for i in ([1, 3, 4, 5] if hidden else [0, 2, 6])]
    if hidden:
        selections += [(p, 6) for p in range(1, len(contract.profiles))]
    return [dict(zip(names(task), (v.tolist() for v in fixtures(contract, contract.profiles[p])[i]))) for p, i in selections]


def validate_input(task, data):
    if task not in RANKS:
        raise ValueError("No active programming task")
    if not isinstance(data, dict) or set(data) != set(names(task)):
        raise ValueError("Input must contain exactly: " + ", ".join(names(task)))
    arrays = []
    for name, rank in zip(names(task), RANKS[task]):
        arr = np.asarray(data[name])
        if arr.dtype.kind not in "iu" or arr.ndim != rank or not all(arr.shape) or arr.size > 1024:
            raise ValueError(f"{name} must be a nonempty integer array with {rank} dimensions and at most 1,024 values")
        if np.any(arr < (0 if task == "smooth" else -512)) or np.any(arr > 4096):
            raise ValueError("Values are outside the task's input range")
        arrays.append(arr.astype(np.int64))
    if task == "practice" and arrays[0].shape != arrays[1].shape:
        raise ValueError("x and y must have the same shape")
    if task in ("threshold_energy", "distances") and arrays[0].shape[1] != arrays[1].shape[-1]:
        raise ValueError("Channel/coordinate dimensions do not match")
    if task == "distances" and arrays[0].shape[0] * arrays[1].shape[0] > 4096:
        raise ValueError("The distance output must contain at most 4,096 values")
    if task == "pool" and (arrays[0].shape[1] % 2 or arrays[0].shape[2] % 2):
        raise ValueError("Image height and width must be even")
    return arrays


def answer(task, data):
    values = validate_input(task, data)
    if task == "practice":
        return (values[0] + values[1]).tolist()
    contract = next(t for t in TASKS if t.name == task)
    x = values[0]
    profile = {
        "calibrate": lambda: {"N": len(x)},
        "background": lambda: dict(zip(("T", "C"), x.shape)),
        "threshold_energy": lambda: dict(zip(("N", "C"), x.shape)),
        "distances": lambda: {"N": x.shape[0], "M": values[1].shape[0], "D": x.shape[1]},
        "pool": lambda: dict(zip(("B", "H", "W", "C"), x.shape)),
        "smooth": lambda: {"H": x.shape[0], "W": x.shape[1], "R": 3},
    }[task]()
    return oracle(contract, profile, copy.deepcopy(values)).tolist()


def task_payload(task, condition):
    public = cases(task)
    return {"id": task, "titles": TITLES[task], "descriptions": DESCRIPTIONS[task],
            "hints": HINTS[task][0], "manual_topics": HINTS[task][1],
            "starter": starter(task, condition), "examples": [
                {"name": f"Case {i+1}", "input": data, "expected": answer(task, data)} for i, data in enumerate(public)],
            "ranks": RANKS[task], "output_rank": OUT_RANK[task], "arguments": names(task)}

# Each assigned ID permanently identifies its task contract. New assignments
# use the latest version, while all saved links retain their original tasks.
from ..revision2 import content as revision2
from ..revision3 import content as revision3
from ..revision4 import content as revision4
from . import practice_v2
from ..revision5 import content as revision5
_legacy_names, _legacy_starter = names, starter
_legacy_cases, _legacy_validate_input = cases, validate_input
_legacy_answer, _legacy_payload = answer, task_payload
VERSIONS = {task: version for version in (revision2, revision3, revision4, revision5, practice_v2) for task in version.RANKS}
ORDER = revision5.ORDER
BUDGETS = revision5.BUDGETS
for version in (revision2, revision3, revision4, revision5, practice_v2):
    TITLES.update(version.TITLES)
    RANKS.update(version.RANKS)
    OUT_RANK.update(version.OUT_RANK)

def names(task):
    return VERSIONS[task].names(task) if task in VERSIONS else _legacy_names(task)

def starter(task, condition):
    return VERSIONS[task].starter(task, condition) if task in VERSIONS else _legacy_starter(task, condition)

def cases(task, hidden=False):
    return VERSIONS[task].cases(task, hidden) if task in VERSIONS else _legacy_cases(task, hidden)

def validate_input(task, data):
    return VERSIONS[task].validate_input(task, data) if task in VERSIONS else _legacy_validate_input(task, data)

def answer(task, data):
    return VERSIONS[task].answer(task, data) if task in VERSIONS else _legacy_answer(task, data)

def task_payload(task, condition):
    return VERSIONS[task].task_payload(task, condition) if task in VERSIONS else _legacy_payload(task, condition)


PRACTICE_IDS = ("practice", "practice_zinnia", "practice_rust")
def is_practice(task):
    return task in PRACTICE_IDS or task in practice_v2.IDS

_versioned_names, _versioned_starter = names, starter
_versioned_cases, _versioned_validate = cases, validate_input
_versioned_answer, _versioned_payload = answer, task_payload
for practice_id in PRACTICE_IDS[1:]:
    TITLES[practice_id] = TITLES["practice"]
    RANKS[practice_id] = [1,1]
    OUT_RANK[practice_id] = 1

def names(task):
    return _versioned_names("practice" if task in PRACTICE_IDS else task)

def starter(task, condition):
    return _versioned_starter("practice" if task in PRACTICE_IDS else task,condition)

def cases(task, hidden=False):
    return _versioned_cases("practice" if task in PRACTICE_IDS else task,hidden)

def validate_input(task, data):
    return _versioned_validate("practice" if task in PRACTICE_IDS else task,data)

def answer(task, data):
    return _versioned_answer("practice" if task in PRACTICE_IDS else task,data)

def task_payload(task, condition):
    result=_versioned_payload("practice" if task in PRACTICE_IDS else task,condition)
    result["id"]=task
    return result
