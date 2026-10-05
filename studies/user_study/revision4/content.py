"""Only the changed filter contract has a new ID; other tasks remain version 3."""
import numpy as np
from ..revision3 import content as previous
from .tasks import FILTER, FILTER_TASK, fixtures, signatures, oracle

ORDER=[FILTER if t=='strided_filters_v3' else t for t in previous.ORDER]
BUDGETS=dict.fromkeys(ORDER,600)
TITLES={FILTER:['Image filters with stride 2 and dilation 3','步长 2、膨胀率 3 的图像滤波']}
RANKS={FILTER:[4,4]}
OUT_RANK={FILTER:4}
DESCRIPTIONS = {'strided_filters_v4': ['Inputs: x has shape [B,H,W,C]: B images, height H, width W, and C values '
                        '(channels) at each pixel. filters has shape [K,U,V,C]: K filters, each with U '
                        'rows, V columns, and one coefficient per channel. The same filters are used '
                        'for every image.\n'
                        '\n'
                        '1. Read B,H,W,C from x and K,U,V,C from filters. Calculate the covered height '
                        'EH = 3*(U-1)+1 and width EW = 3*(V-1)+1. The input guarantees that these fit '
                        'inside the image. Calculate OH = (H-EH)//2+1 and OW = (W-EW)//2+1, where // '
                        'means integer division rounded down. Allocate output [B,OH,OW,K].\n'
                        '2. For each image b and output position (i,j), take a window starting at image '
                        'row 2*i and column 2*j. Select U rows spaced 3 apart and V columns spaced 3 '
                        'apart, keeping all C channels. The window has shape [U,V,C]. Its entry [u,v,c] '
                        'is x[b,2*i+3*u,2*j+3*v,c].\n'
                        '3. For each filter k, multiply this window elementwise by filters[k], also '
                        'shape [U,V,C]. Sum all the products across rows, columns, and channels. Store '
                        'that single total in output[b,i,j,k]. Repeat for all images, output positions, '
                        'and filters, then return output.\n'
                        '4. To extract a window with slices, use row start 2*i, exclusive stop 2*i+EH, '
                        'step 3; use column start 2*j, exclusive stop 2*j+EW, step 3. These bounds stay '
                        'inside the image. Do not pad or reverse the filter. Starting positions advance '
                        'by 2; positions within a window advance by 3.\n'
                        '\n'
                        'You can follow these steps with array operations. A pure loop implementation '
                        'of the same calculation is also accepted.\n'
                        '\n'
                        'Small example with one row and one channel: image [1,2,3,4,5,6], filter [2,3]. '
                        'Starting at column 0, sample values 1 and 4: 1*2 + 4*3 = 14. Move the start to '
                        'column 2 and sample 3 and 6: 3*2 + 6*3 = 24. The output row is [14,24].',
                        '输入：x 的形状为 [B,H,W,C]，表示 B 张图像，每张高 H、宽 W，每个像素有 C 个通道值。filters 的形状为 [K,U,V,C]，表示 '
                        'K 个滤波器，每个有 U 行、V 列，每个位置为各通道提供一个系数。所有图像共用这些滤波器。\n'
                        '\n'
                        '1. 从 x 读取 B、H、W、C，从 filters 读取 K、U、V、C。计算覆盖高度 EH = 3*(U-1)+1、宽度 EW = '
                        '3*(V-1)+1，输入保证它们不超过图像大小。计算 OH = (H-EH)//2+1、OW = (W-EW)//2+1，其中 // '
                        '表示除法后向下取整。创建输出 [B,OH,OW,K]。\n'
                        '2. 对每张图像 b、每个输出位置 (i,j)，从图像第 2*i 行、第 2*j 列开始取窗口。行方向每隔 3 个位置取一行，共取 U 行；列方向每隔 3 '
                        '个位置取一列，共取 V 列；保留所有 C 个通道。窗口形状为 [U,V,C]，其中 [u,v,c] 对应 x[b,2*i+3*u,2*j+3*v,c]。\n'
                        '3. 对每个滤波器 k，将窗口与同为 [U,V,C] 的 filters[k] 逐元素相乘，再将所有行、列、通道的乘积相加，得到一个数，存入 '
                        'output[b,i,j,k]。处理全部图像、输出位置和滤波器后返回 output。\n'
                        '4. 如果使用切片，行起点为 2*i，终点为 2*i+EH（不含），步长为 3；列起点为 2*j，终点为 2*j+EW（不含），步长为 '
                        '3。这些边界不会超出图像。不填充、不翻转滤波器。窗口起点每次移动 2，窗口内部每次取相隔 3 的位置。\n'
                        '\n'
                        '可以按以上步骤使用数组操作，也接受计算结果相同的纯循环实现。\n'
                        '\n'
                        '单行、单通道的小例子：图像为 [1,2,3,4,5,6]，滤波器为 [2,3]。从第 0 列开始，取值 1 和 4，得到 1*2 + 4*3 = '
                        '14。将起点移到第 2 列，取值 3 和 6，得到 3*2 + 6*3 = 24。因此输出行为 [14,24]。']}
HINTS={FILTER:([
 'Calculate the effective window extent and output dimensions first. You can extract each window using step-3 slices, accumulate filter offsets while stepping output starts by 2, or use a sliding-window view. Keep the output step and the spacing within a filter separate. A straightforward implementation using only loops (nested where needed) is also accepted.',
 '先计算实际窗口范围和输出大小。可以用步长为 3 的切片提取窗口，按滤波器偏移累加并将输出起点每次移动 2，或使用滑动窗口视图。分别维护输出步长与滤波器内部的采样间距。也接受最直接的纯循环实现，需要时使用嵌套循环。'],['windows','slicing','sum','matmul'])}
REQUIREMENTS = {'strided_filters_v4': [['1. Read B,H,W,C from x and K,U,V,C from filters. Calculate the covered height '
                         'EH = 3*(U-1)+1 and width EW = 3*(V-1)+1. The input guarantees that these fit '
                         'inside the image. Calculate OH = (H-EH)//2+1 and OW = (W-EW)//2+1, where // '
                         'means integer division rounded down. Allocate output [B,OH,OW,K].'],
                        ['1. 从 x 读取 B、H、W、C，从 filters 读取 K、U、V、C。计算覆盖高度 EH = 3*(U-1)+1、宽度 EW = '
                         '3*(V-1)+1，输入保证它们不超过图像大小。计算 OH = (H-EH)//2+1、OW = (W-EW)//2+1，其中 // '
                         '表示除法后向下取整。创建输出 [B,OH,OW,K]。']]}

def names(task):
    assert task==FILTER
    return ['x','filters']

def starter(task,condition):
    assert task==FILTER
    return previous.starter('strided_filters_v3',condition)

def cases(task,hidden=False):
    assert task==FILTER
    selected=[(0,i) for i in ([1,3,4,5,7] if hidden else [0,2,6])]
    if hidden:selected += [(p,i) for p in range(1,len(FILTER_TASK.profiles)) for i in (2,6)]
    return [dict(zip(names(task),[v.tolist() for v in fixtures(FILTER_TASK,FILTER_TASK.profiles[p])[i]])) for p,i in selected]

def profile(arrays):
    p=dict(zip(('B','H','W','C'),arrays[0].shape))
    p.update(zip(('K','U','V'),arrays[1].shape[:3]))
    return p

def validate_input(task,data):
    assert task==FILTER
    if not isinstance(data,dict) or set(data)!=set(names(task)):raise ValueError('Input must contain exactly: x, filters')
    arrays=[]
    for name in names(task):
        a=np.asarray(data[name])
        if a.dtype.kind not in 'iu' or a.ndim!=4 or not all(a.shape) or a.size>1024 or max(a.shape)>16:
            raise ValueError(f'{name}: expected a nonempty rank-4 integer array, dimensions <=16, at most 1024 values')
        if np.any(a < -16) or np.any(a > 16):raise ValueError(f'{name} values must be in [-16,16]')
        arrays.append(a.astype(np.int64))
    p=profile(arrays)
    if 3*(p['U']-1)+1>p['H'] or 3*(p['V']-1)+1>p['W']:raise ValueError('The dilation-3 filter must fit inside the image')
    shapes,out=signatures(FILTER_TASK,p)
    if any(a.shape!=s for a,s in zip(arrays,shapes.values())):raise ValueError('Input channel dimensions do not match')
    if np.prod(out)>4096:raise ValueError('Output must contain at most 4096 values')
    return arrays

def answer(task,data):
    arrays=validate_input(task,data)
    return oracle(FILTER_TASK,profile(arrays),arrays).tolist()

def task_payload(task,condition):
    return {'id':task,'titles':TITLES[task],'descriptions':DESCRIPTIONS[task],'hints':HINTS[task][0],'manual_topics':HINTS[task][1],
            'requirements':REQUIREMENTS[task],'input_details':[
                'All input values are integers in [-16,16]. Each dimension is positive and at most 16; each input has at most 1,024 values. The dilated filter fits inside the image. Return the result from kernel.',
                '输入均为 [-16,16] 的整数。各维度为 1 到 16，每个输入最多 1,024 个元素。膨胀后的滤波器可以完全放入图像。请从 kernel 返回结果。'],
            'starter':starter(task,condition),'examples':[{'name':f'Case {i+1}','input':v,'expected':answer(task,v)} for i,v in enumerate(cases(task))],
            'ranks':RANKS[task],'output_rank':OUT_RANK[task],'arguments':names(task)}
